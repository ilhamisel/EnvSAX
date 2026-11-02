# Table 4. SAX vs ABBA (v2, participant-wise split)

| band   | method   | param         |   tokens |   compression |     r |
|:-------|:---------|:--------------|---------:|--------------:|------:|
| delta  | SAX      | w=40, a=8     |     40   |          50   | 0.874 |
| delta  | ABBA     | tol=0.05, k=8 |     99.6 |          20.1 | 0.447 |
| delta  | ABBA     | tol=0.1, k=8  |     67.2 |          29.7 | 0.399 |
| delta  | ABBA     | tol=0.2, k=8  |     44.8 |          44.7 | 0.349 |
| delta  | ABBA     | tol=0.4, k=8  |     30.9 |          64.8 | 0.348 |
| theta  | SAX      | w=40, a=8     |     40   |          50   | 0.815 |
| theta  | ABBA     | tol=0.05, k=8 |    131.5 |          15.2 | 0.386 |
| theta  | ABBA     | tol=0.1, k=8  |     88.8 |          22.5 | 0.311 |
| theta  | ABBA     | tol=0.2, k=8  |     58.5 |          34.2 | 0.297 |
| theta  | ABBA     | tol=0.4, k=8  |     41.1 |          48.7 | 0.264 |
| alpha  | SAX      | w=40, a=8     |     40   |          50   | 0.797 |
| alpha  | ABBA     | tol=0.05, k=8 |    147.5 |          13.6 | 0.366 |
| alpha  | ABBA     | tol=0.1, k=8  |     99.7 |          20.1 | 0.341 |
| alpha  | ABBA     | tol=0.2, k=8  |     65.1 |          30.7 | 0.305 |
| alpha  | ABBA     | tol=0.4, k=8  |     45.8 |          43.6 | 0.254 |
| beta   | SAX      | w=40, a=8     |     40   |          50   | 0.688 |
| beta   | ABBA     | tol=0.05, k=8 |    275   |           7.3 | 0.288 |
| beta   | ABBA     | tol=0.1, k=8  |    177.6 |          11.3 | 0.266 |
| beta   | ABBA     | tol=0.2, k=8  |    115.2 |          17.4 | 0.215 |
| beta   | ABBA     | tol=0.4, k=8  |     68.4 |          29.2 | 0.169 |
| gamma  | SAX      | w=40, a=8     |     40   |          50   | 0.695 |
| gamma  | ABBA     | tol=0.05, k=8 |    262.5 |           7.6 | 0.299 |
| gamma  | ABBA     | tol=0.1, k=8  |    169.9 |          11.8 | 0.213 |
| gamma  | ABBA     | tol=0.2, k=8  |    110.9 |          18   | 0.173 |
| gamma  | ABBA     | tol=0.4, k=8  |     66.9 |          29.9 | 0.162 |

## k sweep (alpha)

| method   | param         |   tokens |     r | r_drift   |
|:---------|:--------------|---------:|------:|:----------|
| SAX      | w=40, a=8     |     40   | 0.795 | -         |
| ABBA     | tol=0.1, k=8  |     99.4 | 0.329 | 0.396     |
| ABBA     | tol=0.1, k=16 |     99.4 | 0.43  | 0.501     |
| ABBA     | tol=0.1, k=32 |     99.4 | 0.616 | 0.688     |
| ABBA     | tol=0.1, k=64 |     99.4 | 0.748 | 0.804     |
| ABBA     | tol=0.2, k=8  |     65.3 | 0.306 | 0.355     |
| ABBA     | tol=0.2, k=16 |     65.3 | 0.394 | 0.481     |
| ABBA     | tol=0.2, k=32 |     65.3 | 0.569 | 0.642     |
| ABBA     | tol=0.2, k=64 |     65.3 | 0.691 | 0.755     |
