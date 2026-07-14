import { NextResponse } from "next/server";
import { requireSessionUser } from "../../../../../lib/session";
import { errorMessage, proxyToRag } from "../../../../../lib/rag-server";

export const runtime = "nodejs";

type RouteContext = { params: { documentId: string } };

export async function POST(request: Request, context: RouteContext) {
  try {
    await requireSessionUser();
    const { searchParams } = new URL(request.url);
    const show = searchParams.get("show") ?? "5";
    const { response, payload } = await proxyToRag(
      `/documents/${context.params.documentId}/ingest?show=${show}`,
      { method: "POST" }
    );
    if (!response.ok) {
      return NextResponse.json({ detail: errorMessage(payload, response.status) }, { status: response.status });
    }
    return NextResponse.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Ingestion failed.";
    const status = message === "Authentication required." ? 401 : 500;
    return NextResponse.json({ detail: message }, { status });
  }
}
