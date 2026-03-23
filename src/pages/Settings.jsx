import { useState } from "react";
import { Save, Key, Cpu, Shield } from "lucide-react";

const inputClass =
  "w-full rounded-xl border border-border bg-black/10 px-3 py-2.5 text-sm text-text-primary transition-colors focus:border-accent focus:outline-none";

function Section({ icon: Icon, title, children }) {
  return (
    <div className="panel-soft overflow-hidden rounded-[24px]">
      <div className="flex items-center gap-2.5 border-b border-border bg-black/10 px-5 py-4">
        <Icon size={15} className="text-text-muted" />
        <h2 className="text-sm font-medium text-text-primary">{title}</h2>
      </div>
      <div className="p-5 space-y-5">{children}</div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-text-muted">{label}</label>
      {children}
    </div>
  );
}

export default function Settings() {
  const [settings, setSettings] = useState({
    apiKey: "",
    researcherModel: "haiku",
    coderModel: "sonnet",
    writerModel: "sonnet",
    maxRetries: 3,
    maxPapers: 5,
    dockerTimeout: 60,
    sandboxImage: "research-sandbox:latest",
  });
  const [saved, setSaved] = useState(false);

  const update = (key, value) =>
    setSettings((s) => ({ ...s, [key]: value }));

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-2xl space-y-8 animate-slide-in">
      <div>
        <p className="section-kicker">Configuration</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
          Settings
        </h1>
        <p className="mt-2 text-sm leading-7 text-text-secondary">
          Configure models, sandbox, and API credentials.
        </p>
      </div>

      {/* API Key */}
      <Section icon={Key} title="API Credentials">
        <Field label="Anthropic API Key">
          <input
            type="password"
            value={settings.apiKey}
            onChange={(e) => update("apiKey", e.target.value)}
            placeholder="sk-ant-..."
            className={inputClass}
          />
          <p className="text-[11px] text-text-muted mt-1">
            Stored locally in .env — never committed to git.
          </p>
        </Field>
      </Section>

      {/* Models */}
      <Section icon={Cpu} title="Agent Models">
        <div className="grid gap-4 sm:grid-cols-2">
          {[
            { key: "researcherModel", label: "Researcher Agent" },
            { key: "coderModel", label: "Coder Agent" },
            { key: "writerModel", label: "Writer Agent" },
          ].map(({ key, label }) => (
            <Field key={key} label={label}>
              <select
                value={settings[key]}
                onChange={(e) => update(key, e.target.value)}
                className={`${inputClass} appearance-none cursor-pointer`}
              >
                <option value="haiku">Claude 3.5 Haiku</option>
                <option value="sonnet">Claude 3.7 Sonnet</option>
              </select>
            </Field>
          ))}
          <Field label="Max Retries">
            <input
              type="number"
              min={1}
              max={10}
              value={settings.maxRetries}
              onChange={(e) =>
                update("maxRetries", parseInt(e.target.value) || 3)
              }
              className={inputClass}
            />
          </Field>
          <Field label="Max Papers to Fetch">
            <input
              type="number"
              min={1}
              max={20}
              value={settings.maxPapers}
              onChange={(e) =>
                update("maxPapers", parseInt(e.target.value) || 5)
              }
              className={inputClass}
            />
          </Field>
        </div>
      </Section>

      {/* Sandbox */}
      <Section icon={Shield} title="Docker Sandbox">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Sandbox Image">
            <input
              type="text"
              value={settings.sandboxImage}
              onChange={(e) => update("sandboxImage", e.target.value)}
              className={`${inputClass} font-mono`}
            />
          </Field>
          <Field label="Execution Timeout (seconds)">
            <input
              type="number"
              min={10}
              max={300}
              value={settings.dockerTimeout}
              onChange={(e) =>
                update("dockerTimeout", parseInt(e.target.value) || 60)
              }
              className={inputClass}
            />
          </Field>
        </div>
      </Section>

      {/* Save */}
      <button
        onClick={handleSave}
        className={`flex cursor-pointer items-center gap-2 rounded-2xl px-6 py-3 text-sm font-semibold transition-all ${
          saved
            ? "bg-success-dim text-success"
            : "bg-[linear-gradient(135deg,#67e8f9,#34d399)] text-surface-0 hover:-translate-y-0.5 hover:brightness-110"
        }`}
      >
        <Save size={16} />
        {saved ? "Saved!" : "Save Settings"}
      </button>
    </div>
  );
}
