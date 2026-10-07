export const defaults = Object.freeze({prompt:'A glass observatory above the clouds', negative_prompt:'', seed:42, width:1024, height:1024, steps:30, guidance:7});

// Mirrors GenerationSettings. These settings can be handed to the real local CLI.
export function validate(s) {
  if (typeof s.prompt !== 'string' || !s.prompt.trim()) throw Error('Describe an image with a non-empty prompt.');
  if (typeof s.negative_prompt !== 'string') throw Error('Negative prompt must be text.');
  if ([...s.prompt].length > 2000 || [...s.negative_prompt].length > 2000) throw Error('Prompts must be at most 2,000 characters.');
  for (const [key,min,max] of [['seed',0,4294967295],['steps',1,100]]) {
    if (!Number.isInteger(s[key]) || s[key] < min || s[key] > max) throw Error(`${key} must be an integer from ${min} to ${max}.`);
  }
  for (const key of ['width','height']) if (!Number.isInteger(s[key]) || s[key] < 512 || s[key] > 1536 || s[key] % 64) throw Error(`${key} must be 512–1536 in multiples of 64.`);
  if (s.width*s.height > 1572864) throw Error('Image area must not exceed 1.5 megapixels.');
  if (typeof s.guidance !== 'number' || !Number.isFinite(s.guidance) || s.guidance < 0 || s.guidance > 20) throw Error('Guidance must be a finite number from 0 to 20.');
  return {...s};
}

export function circles(settings) {
  validate(settings);
  let state = settings.seed >>> 0;
  const next = () => (state = (Math.imul(state,1664525)+1013904223)>>>0) / 4294967296;
  return Array.from({length:18},(_,i) => ({x:Math.round(next()*settings.width), y:Math.round(next()*settings.height), r:Math.round((.06+next()*.25)*Math.min(settings.width,settings.height)), color:['#9ef4d3','#aab6ff','#efacd1'][i%3]}));
}

export function svg(settings) {
  const shapes=circles(settings).map(c=>`<circle cx="${c.x}" cy="${c.y}" r="${c.r}" fill="none" stroke="${c.color}" stroke-width="3"/>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${settings.width}" height="${settings.height}" viewBox="0 0 ${settings.width} ${settings.height}"><title>Prism procedural browser preview, seed ${settings.seed}. No AI inference.</title><rect width="100%" height="100%" fill="#121624"/>${shapes}<rect x="24" y="24" width="300" height="48" rx="8" fill="#121624"/><text x="40" y="54" fill="#9ef4d3" font-family="monospace" font-size="15">PRISM / PROCEDURAL / NO AI</text></svg>`;
}

export function manifest(settings) {
  return {schema_version:1, backend:{kind:'browser-procedural',algorithm:'lcg-circles-v1',model_id:null,description:'No model loaded. Only seed and dimensions affect this SVG. This browser renderer differs from the Python Pillow demo.'}, settings:validate(settings)};
}

const quote = value => `'${String(value).replaceAll("'", "'\\''")}'`;
export function command(s) {
  validate(s);
  return `uv run --frozen prism ${quote(s.prompt)} --demo --negative-prompt ${quote(s.negative_prompt)} --seed ${s.seed} --width ${s.width} --height ${s.height} --steps ${s.steps} --guidance ${s.guidance}`;
}
