# Automotive Signal Range Updates - Summary

## Overview
Updated the automotive data quality assessment framework to use more realistic and comprehensive signal ranges based on real-world automotive variations.

## Key Changes

### 1. Enhanced Range Structure
- **Before**: Single `min`/`max` values
- **After**: Dual-tier validation with `soft_min`/`soft_max` and `hard_min`/`hard_max`

### 2. New Validation Strategy
- **Soft ranges**: Typical operating conditions (for warnings)
- **Hard ranges**: Physically possible limits (for invalidation)

### 3. Updated Signal Ranges
Key improvements include:

#### Engine/Powertrain
- **ENGINE_LOAD**: Now allows >100% for boosted engines (calc load can exceed 100%)
- **FUEL_RAIL_PRESSURE**: Expanded to 25,000 kPa for GDI systems
- **DIESEL_RAIL_PRESSURE**: New signal type for diesel common-rail (up to 200,000 kPa)
- **MAP**: Clarified as absolute pressure with realistic ranges for turbo applications
- **AFR_LAMBDA**: Wider hard ranges to account for transient conditions

#### Vehicle Dynamics
- **STEERING_ANGLE**: Extended to ±1080° for various rack ratios
- **STEERING_TORQUE**: Expanded hard range to ±100 N·m for MDPS systems
- **ACCELERATION**: Wider hard ranges (±50 m/s²) to handle IMU spikes
- **WHEEL_SPEED**: Can exceed vehicle speed during slip/ABS events

#### Pressures
- **BOOST_PRESSURE**: Clarified as gauge pressure with proper vacuum range
- **OIL_PRESSURE**: More realistic ranges for modern engines
- **FUEL_PRESSURE**: Separate handling for low-side and high-pressure systems

#### Temperatures
- **INTAKE_TEMP**: Higher limits to account for heat soak
- **CAT_TEMP**: More realistic catalyst temperature ranges

### 4. Updated Validation Logic
- **Priority Issues**: Focus on hard violations (>1%) and high soft violations (>20%)
- **Quality Scoring**: Weighted scoring based on violation severity
- **Reporting**: Separate tracking of soft vs hard violations

### 5. Benefits
- Reduces false positives from unit mix-ups (kPa vs bar vs psi)
- Accounts for diverse powertrain types (NA/turbo/diesel/hybrid/EV)
- Handles sensor saturations and transient spikes more intelligently
- Provides clearer guidance on data quality issues

## Usage
The updated system automatically uses the new ranges. Users will see:
- Warnings for soft violations (values outside typical ranges)
- Errors for hard violations (physically impossible values)
- More accurate quality scoring and priority issue identification

## Version
Updated from v1.2.0 to v1.3.0 to reflect the significant range validation improvements.
