/** Shared UI primitives — bars, labels, dividers, panel chrome. */

interface BarProps {
  label: string;
  value: number;
  max?: number;
  formatValue?: (v: number) => string;
  /** Optional delta arrow glyph ("▲" / "▼") rendered next to the value,
   *  and its color. Purely presentational — callers compute the delta. */
  arrow?: { glyph: "▲" | "▼"; color: string } | null;
}

export function Bar({ label, value, max = 10, formatValue, arrow }: BarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const display = formatValue ? formatValue(value) : value.toFixed(1);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr auto",
        gap: 6,
        alignItems: "center",
        rowGap: 4,
      }}
    >
      <div
        style={{
          fontSize: "12px",
          fontWeight: 500,
          color: "var(--text)",
          letterSpacing: "0.01em",
          textTransform: "capitalize",
        }}
      >
        {label.replace(/_/g, " ")}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          justifySelf: "end",
        }}
      >
        {arrow && (
          <span aria-hidden="true" style={{ fontSize: "9px", color: arrow.color }}>
            {arrow.glyph}
          </span>
        )}
        <span
          style={{
            fontSize: "10.5px",
            color: "var(--text-mute)",
            fontVariantNumeric: "tabular-nums",
            textAlign: "right",
          }}
        >
          {display}
        </span>
      </div>
      <div
        style={{
          gridColumn: "1 / -1",
          height: 4,
          background: "var(--track)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background:
              "linear-gradient(90deg, color-mix(in srgb, var(--accent) 65%, transparent), var(--accent))",
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: "10px",
        fontWeight: 700,
        color: "var(--text-mute)",
        textTransform: "uppercase",
        letterSpacing: "0.16em",
        fontFamily: "var(--font-ui)",
        marginTop: 14,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

export function Divider() {
  return <div style={{ height: 1, background: "var(--hairline-soft)", margin: "10px 0" }} />;
}

interface ToggleProps {
  enabled: boolean;
  label: string;
  onChange?: (next: boolean) => void;
  disabled?: boolean;
}

export function Toggle({ enabled, label, onChange, disabled }: ToggleProps) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        opacity: disabled ? 0.5 : 1,
        padding: "4px 0",
      }}
    >
      <div style={{ fontSize: "11.5px", color: "var(--text)" }}>{label}</div>
      <button
        onClick={() => !disabled && onChange?.(!enabled)}
        style={{
          width: 36,
          height: 22,
          borderRadius: 999,
          background: enabled ? "var(--accent)" : "rgba(255,255,255,0.14)",
          position: "relative",
          transition: "background 0.2s",
          padding: 0,
          cursor: disabled ? "not-allowed" : "pointer",
        }}
        aria-label={`toggle ${label}`}
        disabled={disabled}
      >
        <div
          style={{
            position: "absolute",
            top: 2,
            left: enabled ? 16 : 2,
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: "#ffffff",
            boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
            transition: "left 0.2s",
          }}
        />
      </button>
    </div>
  );
}

export function PanelShell({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="glass"
      style={{
        borderRadius: 22,
        padding: "18px 18px 20px",
        boxShadow: "0 20px 56px rgba(0,0,0,0.38)",
        width: 272,
        maxHeight: 520,
        overflowY: "auto",
        animation: "msg-in 0.25s ease",
      }}
    >
      {children}
    </div>
  );
}
