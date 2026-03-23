import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import NewResearch from "./pages/NewResearch";
import History from "./pages/History";
import Logs from "./pages/Logs";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <div className="flex min-h-screen w-full flex-col lg:flex-row">
      <Sidebar />
      <main className="min-h-screen flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 sm:py-6 lg:px-8 lg:py-8">
          <div className="panel mesh-bg tech-frame panel-cyber min-h-[calc(100vh-2rem)] p-5 sm:p-7 lg:p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/new" element={<NewResearch />} />
              <Route path="/history" element={<History />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </div>
        </div>
      </main>
    </div>
  );
}
