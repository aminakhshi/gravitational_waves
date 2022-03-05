# Codes for: "A template-free approach for waveform extraction of gravitational wave events"

This directory contains the core codes for the paper: "A template-free approach for waveform extraction of gravitational wave events". DOI: [doi.org/10.1038/s41598-021-98821-z](https://doi.org/10.1038/s41598-021-98821-z)

## Description

This work generally suits the condition where one intends to extract signals (bursts) from a highly noisy data. We assume an input time series y(t) which consists of a meaningful signal x(t), additively mixed with some sort of noise n(t):
y(t) = x(t) + n(t).
Our approach is based on a series of noise reduction steps (in a specific order), including noise estimation via minimum statistics through moving windows, which would often result in a near-clear extraction of the burried desired signal x(t).

### Dependencies

Following packages would be necessary to run the codes:

* [Lalsuite](https://pypi.org/project/lalsuite/)
* [PyCBC](https://pypi.org/project/PyCBC/)

Note: To properly install and use these packages visit the associated hyperlinks.

## Authors

* Amin Akhshi (amin.akhshi@gmail.com)
* Hamidreza Alimohammadi (alimohammadi.hamidreza@gmail.com)

## Version History

* 0.1
    * Initial Release

## Acknowledgments

Underlying packages and repositories used in this work should be referenced separately and properly. 

* This research has made use of data or software obtained from the Gravitational Wave Open Science Center (gw-openscience.org), a service of LIGO Laboratory, the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA. LIGO Laboratory and Advanced LIGO are funded by the United States National Science Foundation (NSF) as well as the Science and Technology Facilities Council (STFC) of the United Kingdom, the Max-Planck-Society (MPS), and the State of Niedersachsen/Germany for support of the construction of Advanced LIGO and construction and operation of the GEO600 detector. Additional support for Advanced LIGO was provided by the Australian Research Council. Virgo is funded, through the European Gravitational Observatory (EGO), by the French Centre National de Recherche Scientifique (CNRS), the Italian Istituto Nazionale di Fisica Nucleare (INFN) and the Dutch Nikhef, with contributions by institutions from Belgium, Germany, Greece, Hungary, Ireland, Japan, Monaco, Poland, Portugal, Spain. The construction and operation of KAGRA are funded by Ministry of Education, Culture, Sports, Science and Technology (MEXT), and Japan Society for the Promotion of Science (JSPS), National Research Foundation (NRF) and Ministry of Science and ICT (MSIT) in Korea, Academia Sinica (AS) and the Ministry of Science and Technology (MoST) in Taiwan.
* R. Abbott et al. (LIGO Scientific Collaboration and Virgo Collaboration), "Open data from the first and second observing runs of Advanced LIGO and Advanced Virgo", [SoftwareX 13 (2021) 100658](https://doi.org/10.1016/j.softx.2021.100658).

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This repository is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

To reference this code, please cite the research article: DOI: [doi.org/10.1038/s41598-021-98821-z](https://doi.org/10.1038/s41598-021-98821-z). 

