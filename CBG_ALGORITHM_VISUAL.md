# Constraint-Based Geolocation (CBG) - Visual Guide

This document provides visual explanations and examples of how the CBG algorithm works.

---

## The Big Picture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CBG ALGORITHM OVERVIEW                             │
└──────────────────────────────────────────────────────────────────────┘

Step 1: Measure RTT from Multiple Vantage Points
═══════════════════════════════════════════════════

    VP1 (London)      VP2 (Frankfurt)    VP3 (Amsterdam)
        ●                   ●                  ●
        │                   │                  │
        │ RTT=12ms          │ RTT=15ms         │ RTT=10ms
        │                   │                  │
        └───────────────────┼──────────────────┘
                            ▼
                         Target ?
                         (Unknown location)


Step 2: Convert RTT to Distance
═══════════════════════════════════════════════════

    RTT = 12ms  →  Distance = (12/2) × 200,000 km/s = 1,200 km
    RTT = 15ms  →  Distance = (15/2) × 200,000 km/s = 1,500 km
    RTT = 10ms  →  Distance = (10/2) × 200,000 km/s = 1,000 km


Step 3: Draw Circles
═══════════════════════════════════════════════════

                    Amsterdam (VP3)
                         ●
                    .----+----.
                 .-'     |     '-.
               .'        |        '.
              /          |          \
             |     r=1000km          |
             |           |           |
    London   ●-----------●-----------● Frankfurt
    (VP1)        Paris?      (VP2)
             |  (Target)      |
             |                |
              \              /
               '.          .'
                 '-._   _-'
                     '-'

    Circle 1: Center=London, radius=1200km
    Circle 2: Center=Frankfurt, radius=1500km
    Circle 3: Center=Amsterdam, radius=1000km


Step 4: Find Intersection
═══════════════════════════════════════════════════

                         ●
                    .----+----.
                 .-'XXXXX|XXXXX'-.
               .'XXX     |     XXX'.
              /XXX       |       XXX\
             |XXX        |        XXX|
             |XXX  ┌─────────┐   XXX|
    ●--------●XXX  │Intersect│   XXX●--------●
             |XXX  │  Area   │   XXX|
             |XXX  └─────────┘   XXX|
              \XXX       |       XXX/
               '.XXX     |     XXX.'
                 '-XXXXXX|XXXXXX-'
                     '-●-'

    The shaded area (XXX) is where ALL circles overlap.
    This is where the target MUST be located!


Step 5: Calculate Centroid
═══════════════════════════════════════════════════

    Intersection points: P1, P2, P3, P4, P5, P6

              P1 ●

         P6 ●         ● P2

              ⊕ ← Centroid (estimated location)

         P5 ●         ● P3

              P4 ●

    Centroid = (avg of P1..P6 latitudes, avg of P1..P6 longitudes)

    Result: (48.85°N, 2.35°E) ≈ Paris

```

---

## Example 1: Simple Case (3 VPs)

### Scenario
Geolocate a target in Paris using 3 VPs.

### Input Data
```
Target IP: 213.225.160.239
True Location: Paris (48.8566°N, 2.3522°E)

VP1: London    (51.5074°N, -0.1278°W) → RTT = 12 ms
VP2: Frankfurt (50.1109°N, 8.6821°E)  → RTT = 15 ms
VP3: Amsterdam (52.3676°N, 4.9041°E)  → RTT = 10 ms
```

### Step-by-Step Calculation

**1. Convert RTT to Distance**
```
VP1: d1 = (12 ms / 2) × 200,000 km/s = 1,200 km
VP2: d2 = (15 ms / 2) × 200,000 km/s = 1,500 km
VP3: d3 = (10 ms / 2) × 200,000 km/s = 1,000 km
```

**2. Create Circles**
```
Circle1: center=(51.51°N, -0.13°W), radius=1200 km
Circle2: center=(50.11°N, 8.68°E),  radius=1500 km
Circle3: center=(52.37°N, 4.90°E),  radius=1000 km
```

**3. Circle Preprocessing**
Check for inclusion:
- Distance(London, Amsterdam) = 357 km
- Circle3 radius (1000 km) > 357 km + Circle1 radius difference? No
- No circles fully contained → Keep all 3

**4. Find Intersections**
Circle pairs:
- (Circle1, Circle2): 2 intersection points
- (Circle1, Circle3): 2 intersection points
- (Circle2, Circle3): 2 intersection points
Total: 6 candidate points

**5. Filter Points**
Keep only points inside ALL circles:
- Point A: Inside C1? Yes. Inside C2? Yes. Inside C3? Yes. → Keep
- Point B: Inside C1? Yes. Inside C2? No. → Discard
- ...
Result: 4 valid points

**6. Calculate Centroid**
```python
lat_centroid = (48.92 + 48.87 + 48.79 + 48.84) / 4 = 48.86°N
lon_centroid = (2.41 + 2.28 + 2.35 + 2.38) / 4 = 2.36°E
```

**7. Calculate Error**
```
Estimated: (48.86°N, 2.36°E)
True:      (48.86°N, 2.35°E)
Error = haversine(estimated, true) = 0.8 km
```

**Result: Excellent accuracy!** (< 1 km error)

---

## Example 2: Circle Inclusion Case

### Scenario
Two VPs, one close to target, one far away.

### Visual
```
                           Far VP
                            ●
                        .---+---.
                    .---    |    ---.
                .---        |        ---.
            .---            |            ---.
        .--- Large Circle   |                ---.
    .---     (r=3000km)     |                    ---.
.---                        |                        ---.
|                           |                            |
|                     Close VP                           |
|                          ●                             |
|                     .----+----.                        |
|                  .-'     |     '-.                     |
|                .'        |        '.                   |
|               /   Small  |          \                  |
|              |   Circle  |           |                 |
|              | (r=500km) |           |                 |
|               \       ⊕ Target      /                  |
|                '.        |        .'                   |
|                  '-.     |     .-'                     |
|                     '----+----'                        |
|                          |                             |
 \                         |                            /
  '-.                      |                        .-'
     '---                  |                    ---'
         '---              |                ---'
             '---          |            ---'
                 '---      |        ---'
                     '--   |    --'
                         '-+-'

Analysis:
- Small circle is ENTIRELY inside large circle
- Large circle provides NO additional constraint
- Solution: Remove large circle, keep only small circle

Result:
- Estimated location = Center of small circle
- Much simpler calculation!
```

---

## Example 3: VP Selection Algorithm

### Scenario
Target has measurements from 50 VPs. Which to use?

### All VPs by RTT
```
Rank  VP ID    RTT (ms)  Physical Distance  Speed Check    Use?
────────────────────────────────────────────────────────────────
1     VP_042      8        100 km            ✓ (800km max)  ✓
2     VP_123     12        150 km            ✓              ✓
3     VP_456     15        180 km            ✓              ✓
4     VP_789     18      5,000 km            ✗ (max=1800)   ✗ REJECT
5     VP_234     20        250 km            ✓              ✓
6     VP_567     22        300 km            ✓              ✓
7     VP_890     25        320 km            ✓              ✓
8     VP_111     28        350 km            ✓              ✓
9     VP_222     30        380 km            ✓              ✓
10    VP_333     32        400 km            ✓              ✓
11    VP_444     35        450 km            ✓              ✓ (if n=11)
...
50    VP_999    250     10,000 km            ✓              ✗ (n=10 limit)
```

**Selected VPs:** Top 10 that pass speed check
**Rejected:** VP_789 (physically impossible RTT)

### Why VP_789 Failed
```
RTT = 18 ms
Max possible distance = (18 / 2) × 200,000 = 1,800 km
Actual distance = 5,000 km

5,000 km > 1,800 km  →  IMPOSSIBLE!

Possible causes:
- VP wrongly geolocated
- Anycast routing
- Long routing path
- Measurement error
```

---

## Example 4: Distance Threshold Analysis

### Scenario
Analyze how VP selection affects accuracy.

### Configuration
```
Target: Paris
VPs Available: 100 (worldwide)
Thresholds to test: 0, 100, 500, 1000, 2000 km
```

### Results

**Threshold = 0 km (Use ALL VPs)**
```
VPs used: 100
Median error: 95 km

Visualization:
Global coverage:
    ●●●●●●●●●●●●●●●●
    ●●●●●● ⊕ ●●●●●●
    ●●●●●●●●●●●●●●●●

Many VPs far away → Large circles → Loose constraint
```

**Threshold = 100 km (Only VPs > 100km away)**
```
VPs used: 92 (8 too close)
Median error: 102 km

Removed close VPs:
    ●●●●●●●●●●●●●●●●
    ●●●●●● ⊕ ●●●●●●
    ●●●●●●●●●●●●●●●●
       (no VPs in center)
```

**Threshold = 500 km (Only VPs > 500km away)**
```
VPs used: 75
Median error: 145 km

Only distant VPs:
    ●●●●●●●●●●●●●●●●
    ●●●●●●   ●●●●●●
    ●●●●●●●●●●●●●●●●
         ⊕
```

**Threshold = 1000 km (Only VPs > 1000km away)**
```
VPs used: 45
Median error: 280 km

Very distant VPs:
    ●●●●●●●●●●●●●●●●
    ●●●●●●     ●●●●
    ●●●●●●●●●●●●●●●
           ⊕

Circles too large → Poor accuracy
```

**Conclusion:** Closest VPs provide best accuracy (but not TOO close)

---

## Example 5: Round-Based Algorithm

### Scenario
Iterative refinement using increasingly selective VP sets.

### Round 1: Use All VPs
```
VPs: 100 VPs (worldwide)
Result: (48.90°N, 2.40°E)
Error: 95 km

        ●●●●●●●●●●●●●●
        ●●●●● ⊕ ●●●●●
        ●●●●●●●●●●●●●●
         Large region
```

### Round 2: Use VPs Close to Round 1 Result
```
VPs: 45 VPs (within 500km of previous estimate)
Result: (48.87°N, 2.36°E)
Error: 42 km

           ●●●●●
           ●●⊕●●
           ●●●●●
        Smaller region
```

### Round 3: Use VPs Close to Round 2 Result
```
VPs: 20 VPs (within 200km of previous estimate)
Result: (48.86°N, 2.35°E)
Error: 15 km

             ●●●
             ●⊕●
             ●●●
         Tight region
```

### Convergence
```
Round 1: 95 km error
Round 2: 42 km error (↓ 56%)
Round 3: 15 km error (↓ 64%)
Round 4: 15 km error (converged)
```

**Benefit:** Progressive refinement improves accuracy

---

## Key Insights from Visualizations

### 1. Circle Intersections are Key
```
1 Circle:   ●━━━━●       Entire circle possible
2 Circles:  ●━╋━●        Line segment possible
3+ Circles: ●╋●          Point/small region
```
**More circles = Better precision**

### 2. VP Distance Matters
```
Close VPs (< 100km):    ○ ⊕ ○       Small circles, tight constraint
Medium VPs (100-500km): ○   ⊕   ○   Medium circles, good balance
Far VPs (> 1000km):     ○     ⊕     Large circles, loose constraint
```
**Medium distance = Optimal**

### 3. Geographic Diversity Helps
```
VPs in one direction:         VPs in all directions:
    ●                             ●
    ●                         ●       ●
    ● → ⊕                         ⊕
    ●                         ●       ●
    ●                             ●

  Elongated region            Tight region
```
**Surround target = Better accuracy**

### 4. RTT Inflation Detection
```
Normal RTT:                   Inflated RTT:
VP ─12ms→ Target              VP ─85ms→ Target
Distance: 150 km              Distance: 8,500 km
Max: 1,200 km ✓               Max: 8,500 km ✓ (but suspicious)

If distance is much less than max, likely routing anomaly!
```

---

## Common Patterns

### Pattern 1: European Target
```
VPs mostly in Europe/US → Good coverage
Typical accuracy: 50-200 km
```

### Pattern 2: Asian Target
```
Fewer VPs in region → Large circles
Typical accuracy: 200-500 km
```

### Pattern 3: African Target
```
Very few VPs → Poor coverage
Typical accuracy: 500-1000 km
```

### Pattern 4: Anycast Detection
```
Same IP, multiple RTT clusters:
- Cluster 1: 5-10 ms (Europe)
- Cluster 2: 80-90 ms (US)
→ Anycast detected! Cannot geolocate single location
```

---

## Summary Table

| Component | Input | Output | Visual Representation |
|-----------|-------|--------|----------------------|
| RTT Measurement | Network probes | Milliseconds | ● ─RTT→ ⊕ |
| Distance Conversion | RTT | Kilometers | RTT × speed / 2 |
| Circle Creation | Distance, VP location | Circle on map | ● with radius |
| Intersection | Multiple circles | Overlap region | Shaded area |
| Centroid | Intersection points | Coordinates | ⊕ center point |

**Result:** Estimated location with quantifiable error!
