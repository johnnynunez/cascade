// ASPIRE learning panel (spec §4.8): the lines the camera panel shows for the camera
// server's /state `learning` key (the robot server's learning.json, schema 1).
// Pure: strings in, strings out. The panel puts each line in a text node; nothing
// here is ever parsed as HTML.
export const MAX_LINES = 8;   // served notes shown (learning.json lists at most 5)
export const MAX_LINE = 240;  // characters per line
const CONTROL = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g;
const text = value => {
  if (typeof value !== 'string') return '';
  const clean = value.replace(CONTROL, ' ').replace(/ {2,}/g, ' ').trim();
  return clean.length > MAX_LINE ? clean.slice(0, MAX_LINE - 1) + '…' : clean;
};
// The heading, one line per served note (`<title> — <tag>`), and the last event's line;
// [] for anything that is not a schema-1 state with a hex library sha.
export function learningLines(learning) {
  if (!learning || typeof learning !== 'object' || Array.isArray(learning) || learning.schema !== 1) return [];
  const sha = typeof learning.library_sha === 'string' ? learning.library_sha : '';
  if (!/^[0-9a-f]{8,64}$/.test(sha)) return [];
  const lines = [`Learned notes · lib ${sha.slice(0, 8)}`];
  for (const note of Array.isArray(learning.served) ? learning.served.slice(0, MAX_LINES) : []) {
    const title = text(note?.title), tag = text(note?.tag);
    if (title) lines.push(text(tag ? `${title} — ${tag}` : title));
  }
  const event = learning.last_event;
  const line = event && typeof event === 'object' ? text(event.line) : '';
  if (line) lines.push(line);
  return lines;
}
