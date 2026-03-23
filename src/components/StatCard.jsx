export default function StatCard({ icon: Icon, label, value, sub, accent = false }) {
  return (
    <div className="panel-soft panel-cyber group relative p-5 transition-all duration-200 hover:-translate-y-0.5 hover:border-border-bright">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
      <div className="flex items-start gap-4">
      <div
        className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ring-1 ring-white/8 ${
          accent ? "bg-accent-dim shadow-[0_0_32px_rgba(103,232,249,0.16)]" : "bg-surface-2"
        }`}
      >
        <Icon
          size={18}
          strokeWidth={1.8}
          className={accent ? "text-accent" : "text-text-muted"}
        />
      </div>
      <div>
        <p className="text-3xl font-semibold tracking-tight text-text-primary">
          {value}
        </p>
        <p className="text-xs text-text-secondary mt-0.5">{label}</p>
        {sub && (
          <p className="text-[11px] text-text-muted font-mono mt-1">{sub}</p>
        )}
      </div>
      </div>
    </div>
  );
}
