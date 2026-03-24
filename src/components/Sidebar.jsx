import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  FlaskConical,
  History,
  ScrollText,
  Settings,
  Zap,
} from "lucide-react";

const links = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/new", icon: FlaskConical, label: "New Research" },
  { to: "/history", icon: History, label: "History" },
  { to: "/logs", icon: ScrollText, label: "Logs" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export default function Sidebar() {
  return (
    <aside className="panel-soft panel-cyber mx-4 mt-4 flex h-auto shrink-0 flex-col sm:mx-6 lg:sticky lg:top-4 lg:ml-4 lg:h-[calc(100vh-2rem)] lg:w-72">
      {/* Brand */}
      <div className="border-b border-border px-5 py-6">
        <div className="flex items-start gap-3">
          <div className="accent-glow mt-0.5 flex h-10 w-10 items-center justify-center rounded-xl bg-accent-dim">
            <Zap size={18} className="text-accent" />
          </div>
          <div className="flex-1">
            <p className="section-kicker">Node 01</p>
            <h1 className="mt-2 text-sm font-semibold tracking-tight text-text-primary leading-none">
              Research Claw
            </h1>
            <p className="mt-2 text-[11px] leading-relaxed text-text-secondary">
              Autonomous pipeline cockpit
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `group flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "nav-active text-text-primary"
                  : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
              }`
            }
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-black/10 ring-1 ring-white/5 transition-all duration-200 group-hover:ring-white/10">
              <Icon size={18} strokeWidth={1.8} />
            </div>
            <div className="flex-1">
              <div>{label}</div>
              <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-text-muted">
                sector
              </div>
            </div>
            <div className="text-[10px] font-mono text-text-muted">
              /{label.slice(0, 2).toUpperCase()}
            </div>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="border-t border-border px-5 py-4">
        <div className="rounded-xl bg-black/10 px-3 py-3 ring-1 ring-white/5">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-success animate-pulse" />
            <span className="text-xs font-mono text-text-muted">
              pipeline idle
            </span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-[10px] font-mono text-text-muted">
            <span>uptime 99.2%</span>
            <span>queue 03</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
