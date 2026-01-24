I want to build an application for physicians to monitor sleep EEG metrics from their patients in real-time. The data comes from the OpenBCI Cyton board and is communicated to a computer via the OpenBCI RFduino dongle. I can utilize the BrainFlow library which is an API built to obtain, parse, and analyze data from supported boards (including the Cyton board). I would like to either use a Python, React, and FastAPI stack, though BrainFlow supports C++, Java, Matlab, Julia, Rust, Typescript, and Swift as well.

My reach goal for this application is that a physician can monitor multiple patients by flipping between their dashboards. Each patient will have a Cyton board that sends data to a local computer (maybe Raspberry Pi for scaleability). This data then needs to reach the database that the application uses so that the physician can access it. 

The application should be designed with modularity in mind with respect to the EEG metrics that physicians can access -- one of the first metrics I want to access is band-limited power and some real-time updating spectrograms. I also eventually want to incorporate a notification system that alerts physicians when one of their patients has a "concerning event" detected in their EEG recording. The alert would then point the physician to the segment of the EEG that is of concern. 

The EEG data should also be stored long-term in case physicians want to pull it for further analysis or to do a long-range study.

For starters, though, the MVP needs to have the following:
- Database for patient sleep EEG recordings
- Functions for calculating metrics band-limited power and displaying spectrograms 
- Dashboard for displaying these metrics
- Ability for physicians to flip between multiple patient dashboards 


