# NotNormal 

This package revolves around the **NotNormal** algorithm, which combines estimation and iteration to automatically extract events from (nano)electrochemical time series data.  The [*notnormal*](./docs/index.html) package includes the [*not_normal*](./docs/notnormal/not_normal.html), *not_normal_gui*, [*results*](./docs/notnormal/results.html) and [*simulate*](./docs/notnormal/simulate.html) modules. 

The [*not_normal*](./docs/notnormal/not_normal.html) module contains the functional form of the algorithm, to be used within preexisting code or without the use of the GUI. 

The *not_normal_gui* module is the GUI realisation of the algorithm along with additional functionality to aid in the extraction process. 

The  [*results*](./docs/notnormal/results.html)  module contains dataclasses used internally and for returning verbose results from the functions in the [*not_normal*](./docs/notnormal/not_normal.html)  module.

The  [*simulate*](./docs/notnormal/simulate.html)  module is a WIP and will simulate a trace based on a set number of parameters which describe the noise, baseline and event profiles.

## GUI Quickstart

 1. Click **Browse** -> upon loading, the Z-score will be calculated automatically based on the sample size
 2. Click **Estimate** -> this will estimate the cutoff and event direction for iteration
 3. Click **Iterate** -> this will produce the final results
 4. Click **Save** -> this will produce a CSV file containing the event profiles

## FAQ

**Q.** What is a Bounds Filter? \
**A.** This will help determine bounding for long tailed events, it is only worth changing if inadequate bounding  is identified on conclusion of iteration.

 **Q.** My estimate is attenuating events too much \
 **A.** Reduce the Estimate Cutoff. 

 **Q.** My estimate is bounding my events badly \
 **A.** Increase the Estimate Cutoff.
 
 **Q.** I want to use a different Threshold Window size \
 **A.** Nice! But be informed, an experiment was conducted to determine this was the ideal size for the estimate.
 
 **Q.** I want to use a lower Z-score \
 **A.** This will produce false positives presuming a perfectly normal sample after transformation. It is a good job the estimated number of false positives can be found in the results window upon completion of iteration. What you do with this information remains to be seen.

 **Q.** REPLACE FACTOR!? REPLACE GAP!? \
 **A.** 8 and 2 were determined to be the most consistent options for baseline determination and adequate attenuation of events during replacement. Feel free to experiment, or leave them. 

**Q.** Do we not require a lower Z-score for the estimate because of the event influence? \
**A.** No, the improved estimate is so accurate in the presence of outliers, it blew my socks off. So, we only need one Z-score for both estimation and iteration.

**Q.** What does the estimate accomplish? \
**A.** The estimate will determine the cutoff for iteration and the direction of the events. Note, the direction is the starting point for iteration and will not be the final result. The final result *always* extracts biphasically. If you do not want events from a specific side, they are conveniently labelled in the results CSV.
