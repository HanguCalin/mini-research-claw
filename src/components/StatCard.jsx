export default function StatCard({ icon: Icon, label, value, sub, accent = false }) {
  return (
    <div className="flex items-start gap-4 p-5 rounded-xl border border-border bg-surface-1 hover:border-border-bright transition-colors">
      <div
        className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
          accent ? "bg-accent-dim" : "bg-surface-2"
        }`}
      >
        <Icon
          size={18}
          strokeWidth={1.8}
          className={accent ? "text-accent" : "text-text-muted"}
        />
      </div>
      <div>
        <p className="text-2xl font-semibold text-text-primary tracking-tight">
          {value}
        </p>
        <p className="text-xs text-text-secondary mt-0.5">{label}</p>
        {sub && (
          <p className="text-[11px] text-text-muted font-mono mt-1">{sub}</p>
        )}
      </div>
    </div>
  );
}
