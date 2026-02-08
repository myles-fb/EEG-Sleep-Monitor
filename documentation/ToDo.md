
* Test MOs detection algorithm using existing .edf files
    * Compare performance with Matlab code
    * At 6s step (1/6 f_s), 2-min window only has 20 samples; 5-min window has 300/6 samples = 50 samples. What about smaller step sizes? What's the tradeoff there besides increased computation?
    * Hold off on spatiotemporal regularization for now; stick with matlab pipeline
    * Do we need dominant_frequency_per_window and dominant_frequency_per_band?
        * Scripts for plotting detection
        * .edf conversion to "streaming data" to simulate realtime performance
            for now, don't worry about dashboard. just create plots like in Matlab code. Port what you can over. Worth seeing if there were any outstanding bugs from Matlab code we can catch this time around.
    * Expect a good amount of bugs 🐛

