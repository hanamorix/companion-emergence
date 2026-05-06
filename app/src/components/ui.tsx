/** Shared UI primitives — bars, labels, dividers, panel chrome. */

interface BarProps {
  label: string;
  value: number;
  max?: number;
  formatValue?: (v: number) => string;
}

export function Bar({ label, value, max = 10, formatValue }: BarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const display = formatValue ? formatValue(value) : value.toFixed(1);
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 36px",
        gap: 6,
        alignItems: "center",
        rowGap: 4,
      }}
    >
      <div
        style={{
          fontSize: "11.5px",
          color: "var(--text)",
          letterSpacing: "0.01em",
          textTransform: "capitalize",
        }}
      >
        {label.replace(/_/g, " ")}
      </div>
      <div
        style={{
          fontSize: "10.5px",
          color: "var(--mauve)",
          fontFamily: "var(--font-disp)",
          textAlign: "right",
        }}
      >
        {display}
      </div>
      <div
        style={{
          gridColumn: "1 / -1",
          height: 2.5,
          background: "var(--ash)",
          borderRadius: 1,
          overflow: "hidden",
          opacity: 0.5,
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: "var(--accent)",
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
        fontSize: "9.5px",
        color: "var(--text-mute)",
        textTransform: "uppercase",
        letterSpacing: "0.12em",
        fontFamily: "var(--font-disp)",
        marginTop: 14,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

export function Divider() {
  return <div style={{ height: 1, background: "var(--border)", opacity: 0.3, margin: "10px 0" }} />;
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
          width: 28,
          height: 16,
          borderRadius: 8,
          background: enabled ? "var(--accent)" : "var(--ash)",
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
            left: enabled ? 14 : 2,
            width: 12,
            height: 12,
            borderRadius: "50%",
            background: "var(--linen)",
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
      className="shoji"
      style={{
        background: "var(--panel-bg)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "14px 14px 16px",
        boxShadow: "0 2px 12px rgba(42,31,31,0.18), 0 0 0 1px rgba(130,51,41,0.06)",
        width: 220,
        maxHeight: 360,
        overflowY: "auto",
        animation: "msg-in 0.25s ease",
      }}
    >
      {children}
    </div>
  );
}
