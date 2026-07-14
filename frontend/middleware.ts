import { NextRequest, NextResponse } from "next/server";
import { SESSION_COOKIE, decodeSession } from "./lib/session";

const PUBLIC_PATHS = ["/login", "/signup"];

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isPublicAuthApi =
    pathname.startsWith("/api/auth/login") || pathname.startsWith("/api/auth/signup");
  const isPublicPage = PUBLIC_PATHS.some(
    (path) => pathname === path || pathname.startsWith(`${path}/`)
  );
  const isPublic = isPublicPage || isPublicAuthApi;

  let isAuthed = false;
  try {
    isAuthed = Boolean(await decodeSession(request.cookies.get(SESSION_COOKIE)?.value));
  } catch {
    isAuthed = false;
  }

  if (!isAuthed && !isPublic && !pathname.startsWith("/_next") && pathname !== "/favicon.ico") {
    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ detail: "Authentication required." }, { status: 401 });
    }
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (isAuthed && (pathname === "/login" || pathname === "/signup")) {
    const home = request.nextUrl.clone();
    home.pathname = "/";
    home.search = "";
    return NextResponse.redirect(home);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"]
};
