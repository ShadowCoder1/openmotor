// Vercel Blob access for the intake workflow (private store; BLOB_READ_WRITE_TOKEN in the environment).
//   node blob.mjs get <pathname> <local file>     download one blob
//   node blob.mjs getdir <prefix> <local dir>     download every blob under a prefix (keeps names after the prefix)
//   node blob.mjs put <local file> <pathname>     upload (overwrites)
//   node blob.mjs del <prefix>                    delete every blob under a prefix
import { get, put, list, del } from '@vercel/blob';
import { createWriteStream, createReadStream, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

const [cmd, a, b] = process.argv.slice(2);

async function all(prefix) {
  const out = [];
  let cursor;
  do {
    const r = await list({ prefix, cursor, limit: 1000 });
    out.push(...r.blobs);
    cursor = r.hasMore ? r.cursor : undefined;
  } while (cursor);
  return out;
}

async function download(pathname, dest) {
  const r = await get(pathname, { access: 'private', useCache: false });
  if (!r || r.statusCode !== 200) throw new Error(`not found: ${pathname}`);
  mkdirSync(dirname(dest), { recursive: true });
  await pipeline(Readable.fromWeb(r.stream), createWriteStream(dest));
}

if (cmd === 'get') {
  await download(a, b);
} else if (cmd === 'getdir') {
  for (const x of await all(a)) {
    const rel = x.pathname.slice(a.length);
    if (!rel || rel.includes('..')) continue;
    await download(x.pathname, join(b, rel));
    console.log(rel, x.size);
  }
} else if (cmd === 'put') {
  const size = statSync(a).size;
  await put(b, createReadStream(a), { access: 'private', addRandomSuffix: false, allowOverwrite: true,
                                       multipart: size > 50 * 1024 * 1024 });
} else if (cmd === 'del') {
  const xs = await all(a);
  if (xs.length) await del(xs.map((x) => x.url));
  console.log('deleted', xs.length);
} else {
  console.error('usage: node blob.mjs get|getdir|put|del ...');
  process.exit(2);
}
