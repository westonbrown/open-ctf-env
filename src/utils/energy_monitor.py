from codecarbon import EmissionsTracker
import logging
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class EnergyMonitor:
    """
    Tracks energy consumption for the 'Green Agent' OpenEnv challenge.
    Wraps CodeCarbon to provide simple start/stop/log functionality.
    """
    
    def __init__(self, project_name: str = "open-ctf-agent", output_dir: str = "outputs/energy"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=self.output_dir,
            output_file="emissions.csv",
            log_level="error", # Keep it quiet
            measure_power_secs=1
        )
        self._last_emission: Optional[float] = None

    def start_step(self):
        """Start tracking for a single inference/training step."""
        try:
            self.tracker.start()
        except Exception as e:
            logger.warning(f"Failed to start energy tracker: {e}")

    def stop_step(self) -> Dict[str, float]:
        """
        Stop tracking and return consumption metrics.
        Returns dict with 'kwh', 'co2_kg'.
        """
        try:
            emissions = self.tracker.stop()
            # codecarbon returns total emissions since start of script usually, 
            # but flush/start/stop behavior depends on config.
            # For simplistic per-step, we rely on the delta if running continuous, 
            # or just raw values if wrapped tightly.
            
            # Note: emissions is just the CO2 value float in some versions, 
            # or we read from the object.
            
            return {
                "co2_kg": emissions,
                "energy_kwh": self.tracker.final_emissions_data.energy_consumed if self.tracker.final_emissions_data else 0.0
            }
        except Exception as e:
            logger.warning(f"Failed to stop energy tracker: {e}")
            return {"co2_kg": 0.0, "energy_kwh": 0.0}

    def log_metric(self, step: int, metric_dict: Dict[str, float]):
        """Logs energy metrics to a friendly format."""
        logger.info(f"Step {step} Energy: {metric_dict['energy_kwh']:.8f} kWh | CO2: {metric_dict['co2_kg']:.8f} kg")
