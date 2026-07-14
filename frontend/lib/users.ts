import { createHash, randomBytes, scryptSync, timingSafeEqual } from "crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import path from "path";

export type UserRecord = {
  email: string;
  name: string;
  passwordHash: string;
  createdAt: string;
};

const CSV_HEADER = "email,name,password_hash,created_at";

export function usersCsvPath() {
  return (
    process.env.USERS_CSV_PATH?.trim() ||
    path.join(process.cwd(), "data", "users.csv")
  );
}

function ensureCsv() {
  const filePath = usersCsvPath();
  const dir = path.dirname(filePath);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  if (!existsSync(filePath)) {
    writeFileSync(filePath, `${CSV_HEADER}\n`, "utf8");
  }
  return filePath;
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      cells.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  cells.push(current);
  return cells;
}

function escapeCsv(value: string) {
  if (/[",\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

export function listUsers(): UserRecord[] {
  const filePath = ensureCsv();
  const lines = readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length <= 1) return [];
  return lines.slice(1).map((line) => {
    const [email, name, passwordHash, createdAt] = parseCsvLine(line);
    return {
      email: email.trim().toLowerCase(),
      name: name.trim(),
      passwordHash: passwordHash.trim(),
      createdAt: createdAt.trim()
    };
  });
}

function writeUsers(users: UserRecord[]) {
  const filePath = ensureCsv();
  const body = users
    .map(
      (user) =>
        [
          escapeCsv(user.email),
          escapeCsv(user.name),
          escapeCsv(user.passwordHash),
          escapeCsv(user.createdAt)
        ].join(",")
    )
    .join("\n");
  writeFileSync(filePath, `${CSV_HEADER}\n${body}${users.length ? "\n" : ""}`, "utf8");
}

export function hashPassword(password: string) {
  const salt = randomBytes(16).toString("hex");
  const hash = scryptSync(password, salt, 64).toString("hex");
  return `${salt}:${hash}`;
}

export function verifyPassword(password: string, stored: string) {
  const [salt, hash] = stored.split(":");
  if (!salt || !hash) return false;
  const next = scryptSync(password, salt, 64);
  const expected = Buffer.from(hash, "hex");
  if (expected.length !== next.length) return false;
  return timingSafeEqual(expected, next);
}

export function findUserByEmail(email: string) {
  const normalized = email.trim().toLowerCase();
  return listUsers().find((user) => user.email === normalized) ?? null;
}

export function createUser(input: {
  email: string;
  name: string;
  password: string;
}) {
  const email = input.email.trim().toLowerCase();
  const name = input.name.trim();
  if (!email || !name || input.password.length < 6) {
    throw new Error("Name, email, and a password of at least 6 characters are required.");
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    throw new Error("Enter a valid email address.");
  }
  if (findUserByEmail(email)) {
    throw new Error("An account with this email already exists.");
  }
  const user: UserRecord = {
    email,
    name,
    passwordHash: hashPassword(input.password),
    createdAt: new Date().toISOString()
  };
  writeUsers([...listUsers(), user]);
  return { email: user.email, name: user.name, createdAt: user.createdAt };
}

export function userIdFromEmail(email: string) {
  return createHash("sha256").update(email).digest("hex").slice(0, 16);
}
