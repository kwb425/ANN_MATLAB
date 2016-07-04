[![tag][a]][1]
[![release][b]][2]
[![download][c]][3]
# Table of Contents <a name="anchor_main"></a>
---
1. [Relationships](#anchor_1) <br></br>
2. [Functionalities](#anchor_2) <br></br>
3. [References](#anchor_3) <br></br>

## Relationships <a name="anchor_1"></a> [up](#anchor_main)
1. analysis.m calls organize.m.
2. organize.m calls extract.m, mfcc\_rev.m, sigmoid.m, etc.

## Functionalities <a name="anchor_2"></a> [up](#anchor_main)
1. analysis.m is main().
2. organize.m organizes data neatly.
3. extract.m extracts all necessary informations.
4. mfcc_rev.m extracts MFCC.
5. sigmoid.m is a logsig function for squashing.
6. trained_net.mat is an already trained network used for time efficiency.

## References <a name="anchor_3"></a> [up](#anchor_main)
Please check Git repository for [latest update][4]

Please send any question to: <kwb425@icloud.com>

<!--Links to addresses, reference Markdowns-->
[1]: https://github.com/kwb425/ANN_MATLAB.git
[2]: https://github.com/kwb425/ANN_MATLAB.git
[3]: https://github.com/kwb425/ANN_MATLAB/releases
[4]: https://github.com/kwb425/ANN_MATLAB.git
<!--Links to images, reference Markdowns-->
[a]: https://img.shields.io/badge/Tag-v1.3-red.svg?style=plastic
[b]: https://img.shields.io/badge/Release-v1.3-green.svg?style=plastic
[c]: https://img.shields.io/badge/Download-Click-blue.svg?style=plastic