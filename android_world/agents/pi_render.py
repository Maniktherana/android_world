# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Formats Pi session activity for a terminal.

Kept free of AndroidWorld imports so the formatting stays independent of the
environment stack.
"""

import textwrap
from typing import Any


_DIM = '\033[2m'
_BOLD = '\033[1m'
_CYAN = '\033[36m'
_GREEN = '\033[32m'
_RED = '\033[31m'
_YELLOW = '\033[33m'
_RESET = '\033[0m'

# Agentsims prints flat "key value" header lines, then an indented tree. The
# headers carry the signal, so the tree is dropped from the terminal view.
_STATUS_COLOR = {
    'accepted': _GREEN, 'matched': _GREEN, 'ok': _GREEN,
    'mismatch': _RED, 'none': _RED, 'error': _RED, 'failed': _RED,
    'unknown': _YELLOW, 'unavailable': _YELLOW,
}
_SKIP_KEYS = frozenset(
    {'artifact', 'accessibility', 'image', 'observe', 'action'}
)
_NOISE_PREFIX = ('started=', 'completed=', 'captured=', 'generation=')
_AGENTSIMS_KEYS = frozenset({
    'observe', 'action', 'dispatch', 'verification', 'accessibility',
    'image', 'context', 'elements', 'artifact', 'device', 'submit',
})


def _clip(text: str, limit: int = 110) -> str:
  text = ' '.join(text.split())
  return text if len(text) <= limit else text[: limit - 1] + '…'


def _tint(word: str) -> str:
  color = _STATUS_COLOR.get(word)
  return f'{color}{word}{_RESET}' if color else word


def _shorten_command(command: str) -> str:
  """Drops the device flag and other constant noise from a shell command."""
  tokens, out = command.split(), []
  skip = False
  for token in tokens:
    if skip:
      skip = False
      continue
    if token in ('-d', '--device'):
      skip = True
      continue
    if token.startswith('--device=') or token in ('--json', 'npx'):
      continue
    out.append(token)
  return _clip(' '.join(out))


def _summarize_args(args: Any) -> str:
  if not isinstance(args, dict) or not args:
    return ''
  if 'command' in args:
    return _shorten_command(str(args['command']))
  for key in ('path', 'file_path', 'pattern', 'query'):
    if key in args:
      return _clip(str(args[key]))
  return _clip(', '.join(f'{k}={v!r}' for k, v in args.items()))


def _summarize_result(text: str, failed: bool) -> str:
  """Collapses one tool result into a single readable line."""
  if not text:
    return f'{_DIM}(no output){_RESET}'
  lines = text.splitlines()
  # Agentsims prints flush-left "key value" headers. Other tools, such as a
  # file read, print prose, so only parse headers when the shape matches.
  headers = [
      ln for ln in lines if ln.strip() and not ln.startswith((' ', '-', '\t'))
  ]
  keys = [ln.partition('  ')[0].strip() for ln in headers]
  if not _AGENTSIMS_KEYS.intersection(keys):
    head = next((ln.strip() for ln in lines if ln.strip()), '')
    extra = f' {_DIM}(+{len(lines) - 1} lines){_RESET}' if len(lines) > 1 else ''
    body = f'{_RED}{_clip(head)}{_RESET}' if failed else f'{_DIM}{_clip(head)}{_RESET}'
    return body + extra

  parts: list[str] = []
  for line in headers:
    key, _, rest = line.partition('  ')
    key, rest = key.strip(), rest.strip()
    if key in ('dispatch', 'verification'):
      # Keep the verdict word, drop the sentence that follows it.
      parts.append(f'{key} {_tint(rest.split()[0])}' if rest else key)
    elif key == 'elements':
      parts.append(_clip(rest.split('  ')[0], 24))
    elif key == 'context':
      app = next(
          (t[4:] for t in rest.split() if t.startswith('app=')), ''
      ).rsplit('.', 1)[-1]
      if app:
        parts.append(app)
    elif key not in _SKIP_KEYS:
      keep = ' '.join(
          t for t in rest.split() if not t.startswith(_NOISE_PREFIX)
      )
      parts.append(_clip(f'{key} {keep}'.strip(), 60))
  if not parts:
    parts = [_clip(headers[0])]
  line = ' · '.join(parts[:8])
  return f'{_RED}{line}{_RESET}' if failed else f'{_DIM}{line}{_RESET}'


def render_session_event(event: dict) -> str:
  """Renders one entry from a Pi session JSONL file."""
  if event.get('type') != 'message':
    return ''
  message = event.get('message') or {}
  role = message.get('role')
  content = message.get('content')
  content = content if isinstance(content, list) else []

  if role == 'assistant':
    out = []
    for part in content:
      if part.get('type') == 'toolCall':
        name = part.get('name', '?')
        args = _summarize_args(part.get('arguments'))
        out.append(f'{_CYAN}▸ {name}{_RESET} {args}\n')
      elif part.get('type') == 'text' and part.get('text', '').strip():
        body = textwrap.fill(
            part['text'].strip(), width=96,
            initial_indent='  ', subsequent_indent='  ',
        )
        out.append(f'{_BOLD}{body}{_RESET}\n')
    return ''.join(out)

  if role == 'toolResult':
    text = '\n'.join(
        c.get('text', '') for c in content if c.get('type') == 'text'
    )
    return f'    {_summarize_result(text, bool(message.get("isError")))}\n'

  return ''


def final_text(event: dict) -> str | None:
  """Returns assistant text for a completed message, else None."""
  if event.get('type') != 'message':
    return None
  message = event.get('message') or {}
  if message.get('role') != 'assistant':
    return None
  content = message.get('content')
  if not isinstance(content, list):
    return None
  if any(c.get('type') == 'toolCall' for c in content):
    return None
  text = '\n'.join(
      c.get('text', '') for c in content if c.get('type') == 'text'
  ).strip()
  return text or None
