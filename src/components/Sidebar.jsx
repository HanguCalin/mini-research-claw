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
    <aside className="w-60 shrink-0 h-screen sticky top-0 bg-surface-1 border-r border-border flex flex-col">
      {/* Brand */}
      <div className="px-5 py-6 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-accent-dim flex items-center justify-center">
            <Zap size={18} className="text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-tight leading-none">
              Research Claw
            </h1>
            <span className="text-[11px] text-text-muted font-mono">
              v0.1.0 — claude edition
            </span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 flex flex-col gap-0.5">
        {links.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                isActive
                  ? "bg-accent-dim text-accent"
                  : "text-text-secondary hover:text-text-primary hover:bg-surface-2"
              }`
            }
          >
            <Icon size={18} strokeWidth={1.8} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
          <span className="text-xs text-text-muted font-mono">
            pipeline idle
          </span>
        </div>
      </div>
    </aside>
  );
}
