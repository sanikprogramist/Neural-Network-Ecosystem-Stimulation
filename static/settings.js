import { fetchSettings, restartSimulation } from './api.js';

export const inputsConfig = [
    { input: 'worldSpeedInput', val: 'worldSpeedValue', isFloat: true, fixed: 2 },
    { input: 'maxSpeedInput', val: 'maxSpeedValue', isFloat: false },
    { input: 'maxAngularVelocityInput', val: 'maxAngularVelocityValue', isFloat: true, fixed: 1 },
    { input: 'globalMutationRateInput', val: 'globalMutationRateValue', isFloat: true, fixed: 3 },
    { input: 'globalMutationStrengthInput', val: 'globalMutationStrengthValue', isFloat: true, fixed: 2 },
    { input: 'weightStdNewNeuronsInput', val: 'weightStdNewNeuronsValue', isFloat: true, fixed: 2 },
    { input: 'startingHerbivoreInput', val: 'startingHerbivoreValue', isFloat: false },
    { input: 'startingPredatorInput', val: 'startingPredatorValue', isFloat: false },
    { input: 'startingPlantInput', val: 'startingPlantValue', isFloat: false },
    { input: 'maxPlantInput', val: 'maxPlantValue', isFloat: false },
    { input: 'plantNutritionValueInput', val: 'plantNutritionValueValue', isFloat: true, fixed: 2 },
    { input: 'plantRegrowthPowerInput', val: 'plantRegrowthPowerValue', isFloat: true, fixed: 1 },
    { input: 'maxPredatorInput', val: 'maxPredatorValue', isFloat: false },
    { input: 'predatorAvgGestationInput', val: 'predatorAvgGestationValue', isFloat: true, fixed: 1 },
    { input: 'predatorGestationStdInput', val: 'predatorGestationStdValue', isFloat: true, fixed: 1 },
    { input: 'predatorMinReproductionSatietyInput', val: 'predatorMinReproductionSatietyValue', isFloat: true, fixed: 1 },
    { input: 'predatorReproductionLossInput', val: 'predatorReproductionLossValue', isFloat: true, fixed: 2 },
    { input: 'predatorEatPercentThresholdInput', val: 'predatorEatPercentThresholdValue', isFloat: false },
    { input: 'predatorFOVInput', val: 'predatorFOVValue', isFloat: true, fixed: 2 },
    { input: 'predatorVisionRangeInput', val: 'predatorVisionRangeValue', isFloat: false },
    { input: 'predatorAvgAgeInput', val: 'predatorAvgAgeValue', isFloat: true, fixed: 1 },
    { input: 'predatorAgeStdInput', val: 'predatorAgeStdValue', isFloat: true, fixed: 1 },
    { input: 'predatorMinAgeReproductionInput', val: 'predatorMinAgeReproductionValue', isFloat: true, fixed: 1 },
    { input: 'predatorsResurrectAfterHerbivoresReachInput', val: 'predatorsResurrectAfterHerbivoresReachValue', isFloat: false },
    { input: 'predatorResurrectionCountInput', val: 'predatorResurrectionCountValue', isFloat: false },
    { input: 'predatorResurrectionRecentCountInput', val: 'predatorResurrectionRecentCountValue', isFloat: false },
    { input: 'predatorResurrectionRandomCountInput', val: 'predatorResurrectionRandomCountValue', isFloat: false },
    { input: 'maxHerbivoreInput', val: 'maxHerbivoreValue', isFloat: false },
    { input: 'herbivoreSatietyLossInput', val: 'herbivoreSatietyLossValue', isFloat: true, fixed: 3 },
    { input: 'herbivoreMaxSatietyInput', val: 'herbivoreMaxSatietyValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreAvgGestationInput', val: 'herbivoreAvgGestationValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreGestationStdInput', val: 'herbivoreGestationStdValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreMinReproductionSatietyInput', val: 'herbivoreMinReproductionSatietyValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreReproductionLossInput', val: 'herbivoreReproductionLossValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreEatPercentThresholdInput', val: 'herbivoreEatPercentThresholdValue', isFloat: false },
    { input: 'herbivoreFOVInput', val: 'herbivoreFOVValue', isFloat: true, fixed: 2 },
    { input: 'herbivoreVisionRangeInput', val: 'herbivoreVisionRangeValue', isFloat: false },
    { input: 'herbivoreAvgAgeInput', val: 'herbivoreAvgAgeValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreAgeStdInput', val: 'herbivoreAgeStdValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreMinAgeReproductionInput', val: 'herbivoreMinAgeReproductionValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreNutritionValueInput', val: 'herbivoreNutritionValueValue', isFloat: true, fixed: 1 },
    { input: 'herbivoreResurrectionCountInput', val: 'herbivoreResurrectionCountValue', isFloat: false },
    { input: 'herbivoreResurrectionRandomCountInput', val: 'herbivoreResurrectionRandomCountValue', isFloat: false },
    { input: 'herbivoreResurrectionRecentCountInput', val: 'herbivoreResurrectionRecentCountValue', isFloat: false }
];

export async function syncSlidersWithBackend() {
    try {
        const settings = await fetchSettings();
        const dataMap = {
            'worldSpeedInput': settings.world_speed_multiplier,
            'maxSpeedInput': settings.max_speed,
            'maxAngularVelocityInput': settings.max_angular_velocity,
            'globalMutationRateInput': settings.global_mutation_rate,
            'globalMutationStrengthInput': settings.global_mutation_strength,
            'weightStdNewNeuronsInput': settings.weight_std_for_new_neurons,
            'startingHerbivoreInput': settings.starting_herbivore,
            'startingPredatorInput': settings.starting_predator,
            'startingPlantInput': settings.starting_plant,
            'maxPlantInput': settings.max_plant,
            'plantNutritionValueInput': settings.plant_nutrition_value,
            'plantRegrowthPowerInput': settings.plant_regrowth_power,
            'maxPredatorInput': settings.max_predator,
            'predatorAvgGestationInput': settings.predator_avg_gestation_time,
            'predatorGestationStdInput': settings.predator_gestation_time_std_dev,
            'predatorMinReproductionSatietyInput': settings.predator_reproduction_minimum_satiety,
            'predatorReproductionLossInput': settings.predator_reproduction_satiety_loss,
            'predatorEatPercentThresholdInput': Math.round(settings.predator_max_percent_satiety_to_eat * 100),
            'predatorFOVInput': settings.predator_FOV,
            'predatorVisionRangeInput': settings.predator_vision_range,
            'predatorAvgAgeInput': settings.predator_avg_age,
            'predatorAgeStdInput': settings.predator_age_std_dev,
            'predatorMinAgeReproductionInput': settings.predator_min_age_to_reproduce,
            'predatorsResurrectAfterHerbivoresReachInput': settings.predators_resurrect_after_herbivores_reach,
            'predatorResurrectionCountInput': settings.predator_resurrection_count,
            'predatorResurrectionRecentCountInput': settings.predator_resurrection_recent_count,
            'predatorResurrectionRandomCountInput': settings.predator_resurrection_random_count,
            'maxHerbivoreInput': settings.max_herbivore,
            'herbivoreSatietyLossInput': settings.herbivore_satiety_loss_factor,
            'herbivoreMaxSatietyInput': settings.herbivore_max_satiety,
            'herbivoreAvgGestationInput': settings.herbivore_avg_gestation_time,
            'herbivoreGestationStdInput': settings.herbivore_gestation_time_std_dev,
            'herbivoreMinReproductionSatietyInput': settings.herbivore_reproduction_minimum_satiety,
            'herbivoreReproductionLossInput': settings.herbivore_reproduction_satiety_loss,
            'herbivoreEatPercentThresholdInput': Math.round(settings.herbivore_max_percent_satiety_to_eat * 100),
            'herbivoreFOVInput': settings.herbivore_FOV,
            'herbivoreVisionRangeInput': settings.herbivore_vision_range,
            'herbivoreAvgAgeInput': settings.herbivore_avg_age,
            'herbivoreAgeStdInput': settings.herbivore_age_std_dev,
            'herbivoreMinAgeReproductionInput': settings.herbivore_min_age_to_reproduce,
            'herbivoreNutritionValueInput': settings.herbivore_nutrition_value,
            'herbivoreResurrectionCountInput': settings.herbivore_resurrection_count,
            'herbivoreResurrectionRandomCountInput': settings.herbivore_resurrection_random_count,
            'herbivoreResurrectionRecentCountInput': settings.herbivore_resurrection_recent_count
        };

        inputsConfig.forEach(cfg => {
            const inputEl = document.getElementById(cfg.input);
            const valueEl = document.getElementById(cfg.val);
            const currentVal = dataMap[cfg.input];
            if (inputEl && valueEl && currentVal !== undefined) {
                inputEl.value = currentVal;
                valueEl.textContent = cfg.isFloat ? currentVal.toFixed(cfg.fixed) : currentVal;
            }
        });
    } catch (err) {
        console.error('Failed to sync settings overlay layout data:', err);
        throw err;
    }
}

export async function commitSettingsFromDOM() {
    const getVal = (id, isFloat) => {
        const val = document.getElementById(id).value;
        return isFloat ? parseFloat(val) : parseInt(val, 10);
    };

    const dataToSend = {
        world_speed_multiplier: getVal('worldSpeedInput', true),
        max_speed: getVal('maxSpeedInput', true),
        max_angular_velocity: getVal('maxAngularVelocityInput', true),
        global_mutation_rate: getVal('globalMutationRateInput', true),
        global_mutation_strength: getVal('globalMutationStrengthInput', true),
        weight_std_for_new_neurons: getVal('weightStdNewNeuronsInput', true),
        starting_herbivore: getVal('startingHerbivoreInput', false),
        starting_predator: getVal('startingPredatorInput', false),
        starting_plant: getVal('startingPlantInput', false),
        max_plant: getVal('maxPlantInput', false),
        plant_size: 6,
        plant_nutrition_value: getVal('plantNutritionValueInput', true),
        plant_regrowth_power: getVal('plantRegrowthPowerInput', true),
        max_predator: getVal('maxPredatorInput', false),
        predator_avg_gestation_time: getVal('predatorAvgGestationInput', true),
        predator_gestation_time_std_dev: getVal('predatorGestationStdInput', true),
        predator_reproduction_minimum_satiety: getVal('predatorMinReproductionSatietyInput', true),
        predator_reproduction_satiety_loss: getVal('predatorReproductionLossInput', true),
        predator_max_percent_satiety_to_eat: getVal('predatorEatPercentThresholdInput', false) / 100,
        predator_FOV: getVal('predatorFOVInput', true),
        predator_vision_range: getVal('predatorVisionRangeInput', false),
        predator_avg_age: getVal('predatorAvgAgeInput', true),
        predator_age_std_dev: getVal('predatorAgeStdInput', true),
        predator_min_age_to_reproduce: getVal('predatorMinAgeReproductionInput', true),
        predators_resurrect_after_herbivores_reach: getVal('predatorsResurrectAfterHerbivoresReachInput', false),
        predator_resurrection_count: getVal('predatorResurrectionCountInput', false),
        predator_resurrection_recent_count: getVal('predatorResurrectionRecentCountInput', false),
        predator_resurrection_random_count: getVal('predatorResurrectionRandomCountInput', false),
        max_herbivore: getVal('maxHerbivoreInput', false),
        herbivore_satiety_loss_factor: getVal('herbivoreSatietyLossInput', true),
        herbivore_max_satiety: getVal('herbivoreMaxSatietyInput', true),
        herbivore_avg_gestation_time: getVal('herbivoreAvgGestationInput', true),
        herbivore_gestation_time_std_dev: getVal('herbivoreGestationStdInput', true),
        herbivore_reproduction_minimum_satiety: getVal('herbivoreMinReproductionSatietyInput', true),
        herbivore_reproduction_satiety_loss: getVal('herbivoreReproductionLossInput', true),
        herbivore_max_percent_satiety_to_eat: getVal('herbivoreEatPercentThresholdInput', false) / 100,
        herbivore_FOV: getVal('herbivoreFOVInput', true),
        herbivore_vision_range: getVal('herbivoreVisionRangeInput', false),
        herbivore_avg_age: getVal('herbivoreAvgAgeInput', true),
        herbivore_age_std_dev: getVal('herbivoreAgeStdInput', true),
        herbivore_min_age_to_reproduce: getVal('herbivoreMinAgeReproductionInput', true),
        herbivore_nutrition_value: getVal('herbivoreNutritionValueInput', true),
        herbivore_resurrection_count: getVal('herbivoreResurrectionCountInput', false),
        herbivore_resurrection_random_count: getVal('herbivoreResurrectionRandomCountInput', false),
        herbivore_resurrection_recent_count: getVal('herbivoreResurrectionRecentCountInput', false)
    };

    return restartSimulation(dataToSend);
}
