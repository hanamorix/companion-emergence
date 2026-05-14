import { useEffect, useRef, useState } from "react";
import { PanelShell } from "../ui";
import { getBridgeCredentials } from "../../bridge";

/** An individual image entry returned by GET /images. */
interface ImageEntry {
  sha: string;
  ext: string;
  first_seen_ts: string;
  first_8_chars: string;
}

interface Props {
  persona: string;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "ok"; images: ImageEntry[] }
  | { kind: "error"; detail: string };

/**
 * GalleryPanel — thumbnail grid of past images shared in chat, with a
 * lightbox overlay for full-size viewing.
 */
export function GalleryPanel({ persona }: Props) {
  const [load, setLoad] = useState<LoadState>({ kind: "loading" });
  const [bridgeUrl, setBridgeUrl] = useState<string | null>(null);
  const [lightboxSha, setLightboxSha] = useState<string | null>(null);

  // Resolve the bridge base URL once for <img> src attributes.  Images
  // are served directly (not JSON) so we can't use bridgeFetch's fetch
  // path — but the URL is stable and the auth token isn't needed for
  // GET with the bridge's per-origin Bearer header (the browser sends
  // the same auth headers because we set VITE_BRIDGE_URL consistently).
  useEffect(() => {
    let cancelled = false;
    getBridgeCredentials(persona).then((c) => {
      if (!cancelled) setBridgeUrl(c.url);
    });
    return () => { cancelled = true; };
  }, [persona]);

  useEffect(() => {
    let cancelled = false;
    setLoad({ kind: "loading" });

    async function loadImages() {
      try {
        const creds = await getBridgeCredentials(persona);
        const headers: HeadersInit = creds.authToken
          ? { Authorization: `Bearer ${creds.authToken}` }
          : {};
        const r = await fetch(`${creds.url}/images?limit=50`, { headers });
        if (!r.ok) throw new Error(`/images ${r.status}`);
        const data = (await r.json()) as ImageEntry[];
        if (!cancelled) setLoad({ kind: "ok", images: data });
      } catch (err) {
        if (!cancelled)
          setLoad({ kind: "error", detail: String(err) });
      }
    }

    loadImages();
    return () => { cancelled = true; };
  }, [persona]);

  // Lightbox: close on Escape
  useEffect(() => {
    if (!lightboxSha) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setLightboxSha(null);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightboxSha]);

  if (load.kind === "loading") {
    return (
      <PanelShell>
        <div style={{ fontSize: "11.5px", color: "var(--text-mute)", padding: "12px 0", textAlign: "center" }}>
          Loading gallery…
        </div>
      </PanelShell>
    );
  }

  if (load.kind === "error") {
    return (
      <PanelShell>
        <div style={{ fontSize: "11.5px", color: "var(--text-mute)", padding: "12px 0", textAlign: "center" }}>
          Could not load images. {load.detail}
        </div>
      </PanelShell>
    );
  }

  if (load.images.length === 0) {
    return (
      <PanelShell>
        <div style={{ fontSize: "11.5px", color: "var(--text-mute)", padding: "20px 8px", textAlign: "center", lineHeight: 1.5 }}>
          No images shared yet.
          <br />
          Send an image in chat to see it here.
        </div>
      </PanelShell>
    );
  }

  const panel = (
    <PanelShell>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 6,
        }}
      >
        {load.images.map((img) => (
          <Thumbnail
            key={img.sha}
            sha={img.sha}
            bridgeUrl={bridgeUrl}
            onClick={() => setLightboxSha(img.sha)}
          />
        ))}
      </div>

      {lightboxSha && bridgeUrl && (
        <Lightbox
          sha={lightboxSha}
          bridgeUrl={bridgeUrl}
          onClose={() => setLightboxSha(null)}
        />
      )}
    </PanelShell>
  );

  return panel;
}

/* ------------------------------------------------------------------ */
/*  Thumbnail                                                         */
/* ------------------------------------------------------------------ */

function Thumbnail({
  sha,
  bridgeUrl,
  onClick,
}: {
  sha: string;
  bridgeUrl: string | null;
  onClick: () => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);

  // Lazy-load via IntersectionObserver
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { rootMargin: "100px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const src = bridgeUrl ? `${bridgeUrl}/images/${sha}` : "";

  return (
    <div
      ref={ref}
      role="button"
      tabIndex={0}
      aria-label={`View image ${sha.slice(0, 8)}`}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onClick();
      }}
      style={{
        aspectRatio: "1 / 1",
        borderRadius: 4,
        overflow: "hidden",
        cursor: "pointer",
        background: "var(--ash)",
        border: "1px solid var(--border)",
        opacity: visible ? 1 : 0.3,
        transition: "opacity 0.2s",
      }}
    >
      {visible && src && (
        <img
          src={src}
          alt={`Image ${sha.slice(0, 8)}`}
          loading="lazy"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Lightbox                                                          */
/* ------------------------------------------------------------------ */

function Lightbox({
  sha,
  bridgeUrl,
  onClose,
}: {
  sha: string;
  bridgeUrl: string;
  onClose: () => void;
}) {
  // Prevent body scroll while lightbox is open
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = prev; };
  }, []);

  return (
    <div
      role="dialog"
      aria-label="Image viewer"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(24, 18, 18, 0.85)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <img
        src={`${bridgeUrl}/images/${sha}`}
        alt={`Full-size image ${sha.slice(0, 8)}`}
        style={{
          maxWidth: "90vw",
          maxHeight: "90vh",
          objectFit: "contain",
          borderRadius: 4,
          boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
        }}
        onClick={(e) => e.stopPropagation()}
      />

      <button
        onClick={onClose}
        aria-label="Close image viewer"
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          width: 32,
          height: 32,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.12)",
          border: "1px solid rgba(255,255,255,0.18)",
          color: "var(--linen)",
          fontSize: 18,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
        }}
      >
        ✕
      </button>
    </div>
  );
}
