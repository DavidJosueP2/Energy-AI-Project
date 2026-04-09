"""Quick integration test for the new fuzzy inference system."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import AppConfig
from app.fuzzy.controller import FuzzyController
from app.fuzzy.linguistic import LinguisticInput, LinguisticOutput
from app.simulation.devices import get_hvac_config, get_refrigerator_config, ControlledDevice

# Test 1: Linguistic input/output
print("=== Test 1: Linguistic Input/Output ===")
li = LinguisticInput()
lo = LinguisticOutput()

selections = {
    'temperatura': 'alta',
    'humedad': 'alta',
    'ocupacion': 'media',
    'tarifa': 'media',
}
crisp = li.to_controller_inputs(selections)
print(f"  Selections: {selections}")
print(f"  Crisp inputs: {crisp}")

# Test 2: Controller with humidity
print("\n=== Test 2: Fuzzy Controller with Humidity ===")
config = AppConfig()
fc = FuzzyController(config.fuzzy)
print(f"  Controller: {fc}")
print(f"  Input variables: {list(fc.input_variables.keys())}")
assert 'humidity' in fc.input_variables, "Humidity variable missing!"

# Test 3: Inference with detail
print("\n=== Test 3: Inference with Detail ===")
output_val, detail = fc.evaluate_with_detail(crisp)
dual = lo.get_dual_output(output_val)
print(f"  Output value: {output_val:.2f}")
print(f"  Output label: {dual['etiqueta_display']}")
print(f"  Active rules: {len(detail.active_rules)}")
if detail.top_rules:
    rule, strength = detail.top_rules[0]
    print(f"  Top rule: {rule} -> strength={strength:.3f}")
print(f"  Aggregated shape: {detail.aggregated_output.shape}")

# Test 4: Devices
print("\n=== Test 4: Controlled Devices ===")
hvac = ControlledDevice(get_hvac_config())
fridge = ControlledDevice(get_refrigerator_config())
print(f"  HVAC: {hvac.display_name}, target={hvac.config.target_temperature}C")
print(f"  Fridge: {fridge.display_name}, target={fridge.config.target_temperature}C")

# Step device
state = hvac.step(35.0, 3, 500, 75.0)
print(f"  HVAC step: {state}")

# Test 5: Simulation with humidity
print("\n=== Test 5: Simulation with Humidity ===")
from app.simulation.simulator import Simulator
config.simulation.horizon_hours = 24
sim = Simulator(config)
result = sim.run(fc.get_controller_function(), label="test")
print(f"  Steps: {result.num_steps}")
print(f"  Has humidity: {'humidity' in result.data.columns}")
print(f"  Humidity range: [{result.data['humidity'].min():.2f}, {result.data['humidity'].max():.2f}]")

# Test 6: Fuzzy plots
print("\n=== Test 6: Fuzzy Plots ===")
from app.visualization import fuzzy_plots
fig = fuzzy_plots.plot_all_membership_functions(
    fc.input_variables, fc.output_variable
)
print(f"  All MF plot: {fig.get_size_inches()}")

fig2 = fuzzy_plots.plot_rule_activations(detail.rules_with_strength)
print(f"  Rule activation plot: {fig2.get_size_inches()}")

fig3 = fuzzy_plots.plot_aggregation_defuzzification(
    fc.output_variable, detail.aggregated_output, detail.centroid_value
)
print(f"  Aggregation plot: {fig3.get_size_inches()}")

import matplotlib.pyplot as plt
plt.close('all')

# Test 7: GA chromosome with humidity
print("\n=== Test 7: GA Chromosome with Humidity ===")
from app.genetic.chromosome import ChromosomeEncoder
encoder = ChromosomeEncoder(config.fuzzy)
print(f"  Chromosome length: {encoder.chromosome_length}")
default = encoder.encode_default()
print(f"  Default chromosome shape: {default.shape}")

print("\n=== ALL TESTS PASSED ===")
