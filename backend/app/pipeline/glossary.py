"""Domain-aware glossary: do-not-translate terms + forced replacements.

Usage:
    g = Glossary.from_domain("finance")
    g = g.merge(Glossary.from_user_text("SIP=SIP\nNifty=Nifty"))
    system_prompt_fragment = g.as_prompt_text()
"""
from __future__ import annotations
from dataclasses import dataclass, field


_GLOSSARY_PRESETS: dict[str, "Glossary"] = {}


@dataclass
class Glossary:
    do_not_translate: list[str] = field(default_factory=list)
    forced: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_domain(cls, domain: str) -> "Glossary":
        return _GLOSSARY_PRESETS.get(domain, cls())

    @classmethod
    def from_user_text(cls, text: str) -> "Glossary":
        """Parse 'Term=Replacement' lines. Lines without '=' are do-not-translate terms."""
        dnt: list[str] = []
        forced: dict[str, str] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                src, _, tgt = line.partition("=")
                forced[src.strip()] = tgt.strip()
            else:
                dnt.append(line)
        return cls(do_not_translate=dnt, forced=forced)

    def merge(self, other: "Glossary") -> "Glossary":
        return Glossary(
            do_not_translate=list(dict.fromkeys(self.do_not_translate + other.do_not_translate)),
            forced={**self.forced, **other.forced},
        )

    def as_prompt_text(self) -> str:
        lines: list[str] = []
        if self.do_not_translate:
            terms = ", ".join(self.do_not_translate)
            lines.append(f"Never translate these terms — keep them exactly as-is: {terms}.")
        if self.forced:
            pairs = "; ".join(f'"{s}" → "{t}"' for s, t in self.forced.items())
            lines.append(f"Always translate these terms as specified: {pairs}.")
        return " ".join(lines)

    def is_empty(self) -> bool:
        return not self.do_not_translate and not self.forced


# ------------------------------------------------------------------ #
# Preset domain glossaries                                            #
# ------------------------------------------------------------------ #
_GLOSSARY_PRESETS.update({
    "finance": Glossary(
        do_not_translate=["SIP", "STP", "SWP", "ELSS", "SEBI", "AMFI", "Nifty", "Sensex",
                          "BSE", "NSE", "NAV", "AUM", "KYC", "PAN", "demat"],
    ),
    "tech": Glossary(
        do_not_translate=["API", "SDK", "UI", "UX", "AI", "ML", "LLM", "GPU", "CPU",
                          "backend", "frontend", "DevOps", "CI/CD"],
    ),
    "spirituality": Glossary(
        do_not_translate=["karma", "dharma", "moksha", "samsara", "atman", "brahman",
                          "puja", "mantra", "ashram", "guru", "yoga", "pranayama"],
    ),
    "sports": Glossary(
        do_not_translate=["IPL", "ICC", "BCCI", "T20", "ODI", "Test", "DRS", "LBW"],
    ),
    "medical": Glossary(
        do_not_translate=["CT", "MRI", "ECG", "ICU", "OPD", "IV", "BP", "BMI",
                          "diabetes", "hypertension"],
    ),
    "legal": Glossary(
        do_not_translate=["FIR", "PIL", "CrPC", "IPC", "HC", "SC", "NGO"],
    ),
})
