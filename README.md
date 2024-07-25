# BandBuddy

BandBuddy is an experimental product that was designed for the 2021 ECE Senior Capstone at Northeastern University. Our team featured members: Ryan Heminway, Sam Paniccia, Walter Galdamez, Rubens Lacouture, and Brickman Malham. Ultimately, our team won 3rd place among all ECE submissions for our work. The judges recognized our innovative combination of hardware and software techniques, deploying machine learning models on a Jetson Nano which, at the time in 2021, was an uncommon feat.  

## About

The inspiration for this project is the practicing guitarist, sitting in their room and longing for the musical support and inspiration of a band. Enter: BandBuddy, a guitar pedal for guitarists and bassists that sought to revolutionize the practice experience for musicians by acting as a creative band partner. This pedal acted similar to typical looper pedals, but would dynamically generate drum tracks to accompany the user. With extensive control over the genre, bpm, and timbre of the generated drums, this tool provides a springboard for the musician to experiment, learn, and enjoy the practice experience. 

Our system leveraged Magenta GrooVAE models for the drum generation, and trained individual models on varying datasets of specific genres to provide user customization of their sound. We deployed these models to run on a Jetson Nano along with custom firmware and a webserver to provide user control, all orchestrated via a Raspberry Pi. Our system was packaged in to a familiar guitar pedal form-factor, with a push button designed to be stepped on, and volume knobs for tactile control. For a more comprehensive overview of the system, please refer to the final report video. Alternatively, you can read our project overview in our [Project Abstract](BandBuddy_Abstract.pdf).

While the code in this repository is quite specific to our project, we have structured it into organized folders of ML, Data, Firmware, and Webserver for review. We hope our work can inspire future individuals who have a similar passion for combining their technical engineering skills with their musical hobbies.

## Demonstrations

Unfortunately, this project was heavily impacted by the Covid-19 pandemic, which prevented us from conducting a conference-style demonstration to our 2021 cohort in-person. In lieu of this, we recorded a presentation which featured a design overview with demonstrations of the product and its capabilities. Although GitHub's free plan does not permit us to upload a video of this size directly, you can view our [BandBuddy Presentation Video Here](https://drive.google.com/file/d/1wapezHUzZOpXx3CwB58EvInzfpDUlrSQ/view?usp=sharing). Skip to 6:45 mark if you are purely interested in the product demonstration (Ryan Heminway using the pedal with his bass). 
