# E1 Evaluation Result

## Data Summary

- Sample rows: 360
- Official gold rows: 344
- Strictly merged evaluation rows: 344
- Excluded rows after strict merge: 16
- Main-set rows: 243
- Difficult-set rows: 101

## Aspect Metrics

### rule_only

- Aspect macro-F1: 0.5107
- Difficult-set Jaccard: 0.7812

### zeroshot_only

- Aspect macro-F1: 0.0932
- Difficult-set Jaccard: 0.0099

### hybrid

- Aspect macro-F1: 0.496
- Difficult-set Jaccard: 0.7911

## Sentiment

- Sentiment macro-F1: 0.4458

## Key Findings

- Hybrid vs rule_only: Aspect macro-F1 is lower, Difficult-set Jaccard is higher.
- Hybrid vs zeroshot_only: Aspect macro-F1 is higher, Difficult-set Jaccard is higher.
- Most confused aspect pairs under hybrid:
  - gold=service -> pred=room_facilities (26)
  - gold=room_facilities -> pred=quiet_sleep (11)
  - gold=room_facilities -> pred=location_transport (9)

## Confusion Matrix

### rule_only

| gold ↓ / pred → | location_transport | cleanliness | service | room_facilities | quiet_sleep | value | general |
|---|---|---|---|---|---|---|---|
| location_transport | 36 | 2 | 0 | 4 | 1 | 0 | 1 |
| cleanliness | 0 | 10 | 0 | 0 | 0 | 0 | 1 |
| service | 4 | 2 | 30 | 26 | 3 | 6 | 3 |
| room_facilities | 9 | 5 | 0 | 25 | 11 | 1 | 5 |
| quiet_sleep | 3 | 0 | 0 | 0 | 12 | 0 | 0 |
| value | 2 | 0 | 0 | 5 | 1 | 12 | 2 |
| general | 1 | 6 | 1 | 5 | 2 | 2 | 4 |

### zeroshot_only

| gold ↓ / pred → | location_transport | cleanliness | service | room_facilities | quiet_sleep | value | general |
|---|---|---|---|---|---|---|---|
| location_transport | 1 | 0 | 0 | 0 | 0 | 0 | 43 |
| cleanliness | 0 | 1 | 0 | 0 | 0 | 0 | 10 |
| service | 0 | 0 | 3 | 0 | 0 | 0 | 71 |
| room_facilities | 0 | 2 | 0 | 3 | 0 | 0 | 51 |
| quiet_sleep | 0 | 0 | 0 | 0 | 0 | 0 | 15 |
| value | 0 | 0 | 0 | 0 | 0 | 2 | 20 |
| general | 0 | 0 | 0 | 0 | 1 | 3 | 17 |

### hybrid

| gold ↓ / pred → | location_transport | cleanliness | service | room_facilities | quiet_sleep | value | general |
|---|---|---|---|---|---|---|---|
| location_transport | 37 | 2 | 0 | 4 | 1 | 0 | 0 |
| cleanliness | 0 | 11 | 0 | 0 | 0 | 0 | 0 |
| service | 4 | 2 | 33 | 26 | 3 | 6 | 0 |
| room_facilities | 9 | 7 | 0 | 28 | 11 | 1 | 0 |
| quiet_sleep | 3 | 0 | 0 | 0 | 12 | 0 | 0 |
| value | 2 | 0 | 0 | 5 | 1 | 14 | 0 |
| general | 1 | 6 | 1 | 5 | 3 | 5 | 0 |

## Representative Error Cases

### rule_only

1. `0dd1db6f8e5e26ee_s004` | gold=['room_facilities', 'service'] | pred=['service']
   Staff was great and accommodations were excellent!!
2. `1475121e84cd022b_s025` | gold=['cleanliness', 'room_facilities'] | pred=['cleanliness']
   It's old and smells a bit musty, but I don't mind that.
3. `1b1fece8ca00ba29_s005` | gold=['location_transport'] | pred=['cleanliness', 'location_transport']
   The beach access is perfect and the beach was clean.
4. `2f9ba7de916cc18c_s005` | gold=['cleanliness', 'room_facilities'] | pred=['cleanliness']
   While the decor wasn't fancy, the hotel was always clean.
5. `325b056dca9a59ff_s004` | gold=['general'] | pred=['cleanliness']
   The fresh baked croissants were an excellent way to start the day.

### zeroshot_only

1. `0dd1db6f8e5e26ee_s004` | gold=['room_facilities', 'service'] | pred=['general']
   Staff was great and accommodations were excellent!!
2. `100c12d8780361ba_s005` | gold=['cleanliness', 'room_facilities'] | pred=['general']
   The room was clean and comfortable.
3. `118e4ecd9e753fb9_s004` | gold=['cleanliness', 'room_facilities', 'service'] | pred=['general']
   Again, very personable helpful assistance at the desk, full breakfast excellent, elevator musty, room not so clean, internet not working, one phone not working.
4. `1475121e84cd022b_s025` | gold=['cleanliness', 'room_facilities'] | pred=['general']
   It's old and smells a bit musty, but I don't mind that.
5. `1b1fece8ca00ba29_s005` | gold=['location_transport'] | pred=['general']
   The beach access is perfect and the beach was clean.

### hybrid

1. `0dd1db6f8e5e26ee_s004` | gold=['room_facilities', 'service'] | pred=['service']
   Staff was great and accommodations were excellent!!
2. `1475121e84cd022b_s025` | gold=['cleanliness', 'room_facilities'] | pred=['cleanliness']
   It's old and smells a bit musty, but I don't mind that.
3. `1b1fece8ca00ba29_s005` | gold=['location_transport'] | pred=['cleanliness', 'location_transport']
   The beach access is perfect and the beach was clean.
4. `2f9ba7de916cc18c_s005` | gold=['cleanliness', 'room_facilities'] | pred=['cleanliness']
   While the decor wasn't fancy, the hotel was always clean.
5. `325b056dca9a59ff_s004` | gold=['general'] | pred=['cleanliness']
   The fresh baked croissants were an excellent way to start the day.

## Notes

- `aspect_sentiment_gold.csv` is the only official E1 gold.
- Merchant-reply rows remain excluded from the final gold set.