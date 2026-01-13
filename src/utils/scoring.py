import logging
import time
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GreenMetric:
    flags_captured: int
    total_steps: int
    total_energy_wh: float
    total_time_s: float
    
    @property
    def wh_per_flag(self) -> float:
        return self.total_energy_wh / max(1, self.flags_captured)
    
    @property
    def score(self) -> float:
        """
        Composite Green Score: Higher is better.
        Formula: (Flags * 100) / (Energy_Wh * Time_Hrs + 1)
        """
        time_hrs = self.total_time_s / 3600.0
        return (self.flags_captured * 100.0) / ((self.total_energy_wh * time_hrs) + 1.0)

class GreenScoreCard:
    """
    Generates a localized 'Sustainability Report' for the agent.
    """
    def __init__(self, agent_name: str = "OpenCTF-Green-v1"):
        self.agent_name = agent_name
        self.metrics = []
    
    def add_run(self, flags: int, steps: int, energy_wh: float, duration_s: float):
        self.metrics.append(GreenMetric(flags, steps, energy_wh, duration_s))
        
    def generate_report(self) -> str:
        if not self.metrics:
            return "No data recorded."
            
        total_flags = sum(m.flags_captured for m in self.metrics)
        total_energy = sum(m.total_energy_wh for m in self.metrics)
        avg_wh_per_flag = total_energy / max(1, total_flags)
        
        # Real-world comparisons (approximate)
        # Charging Smartphone: ~15 Wh
        # Boiling Kettle: ~100 Wh
        # Training LLM (Big): ~100,000,000 Wh
        
        comparisons = []
        if total_energy < 15:
            comparisons.append(f"Less energy than charging a phone ({total_energy:.2f} Wh vs ~15 Wh)")
        elif total_energy < 100:
            comparisons.append(f"Less energy than boiling a kettle ({total_energy:.2f} Wh vs ~100 Wh)")
            
        report = f"""
# 🌿 Green Agent Score Card: {self.agent_name}

## Summary
*   **Total Flags Captured:** {total_flags}
*   **Total Energy Consumed:** {total_energy:.4f} Wh
*   **Efficiency:** {avg_wh_per_flag:.4f} Wh/Flag

## Sustainability Context
{chr(10).join(['* ' + c for c in comparisons])}

## Verdict
This agent demonstrates **{'High' if avg_wh_per_flag < 5 else 'Moderate'} Efficiency** suitable for edge deployment.
"""
        return report

# Example Usage
if __name__ == "__main__":
    card = GreenScoreCard()
    card.add_run(flags=5, steps=500, energy_wh=2.5, duration_s=600)
    print(card.generate_report())
