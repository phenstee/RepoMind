export interface SSEFrame {
  id?: number;
  event: string;
  data: unknown;
}

export class SSEParser {
  private buffer = "";

  feed(chunk: string): SSEFrame[] {
    this.buffer += chunk;
    const frames: SSEFrame[] = [];
    let boundary = this.buffer.search(/\r?\n\r?\n/);
    while (boundary >= 0) {
      const separator = this.buffer.match(/\r?\n\r?\n/)?.[0];
      if (separator === undefined) {
        break;
      }
      const rawFrame = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + separator.length);
      const parsed = parseFrame(rawFrame);
      if (parsed !== null) {
        frames.push(parsed);
      }
      boundary = this.buffer.search(/\r?\n\r?\n/);
    }
    return frames;
  }
}

function parseFrame(rawFrame: string): SSEFrame | null {
  let id: number | undefined;
  let event = "message";
  const dataLines: string[] = [];
  for (const line of rawFrame.split(/\r?\n/)) {
    if (line.startsWith("id:")) {
      const candidate = Number(line.slice(3).trim());
      if (Number.isInteger(candidate) && candidate > 0) {
        id = candidate;
      }
    } else if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  try {
    return { id, event, data: JSON.parse(dataLines.join("\n")) as unknown };
  } catch {
    throw new Error("The server sent an invalid streaming event.");
  }
}
