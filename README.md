# UNIQUE

This repository provides pretrained models and example codes for the paper:

**UNIQUE: UNified, High-resolution Intelligent Carbon QUantification and Explanation** (*In revision, Remote Sensing of Environment*)

Bokyung Son*, Taejun Sung*, Sejeong Bae, Minki Choo, Jungho Im†, Yeonsu Lee, S.M. Sohel Rana, Dongjin Cho, Cheolhee Yoo, Jeonghyun Hong, Hojin Lee, Hyun Seok Kim



UNIQUE is a framework for estimating gross primary productivity (GPP) at high spatiotemporal resolution by integrating MODIS and Landsat satellite observations. The final output of the framework is daily GPP at 30 m spatial resolution.

## Overview

The UNIQUE framework consists of two main parts.

### Part 1: Satellite-specific GPP estimation

Part 1 estimates GPP separately from MODIS- and Landsat-based inputs using tabular datasets.

* **GPPM**: MODIS-based GPP estimate
* **GPPL**: Landsat-based GPP estimate

The main inputs include satellite-derived vegetation indices and environmental variables.

### Part 2: Fusion of Part 1 estimates

Part 2 fuses the Part 1 estimates to generate daily 30 m GPP.

* **GPPUNIQUE**: final high-spatiotemporal-resolution GPP estimate

Part 2 uses image-based inputs derived from GPPM, GPPL, and auxiliary spatial information. The repository includes code related to the fusion models used in the manuscript, including U-SCALER.


## Notes

The current version of this repository is intended to provide access to pretrained models and example codes associated with the manuscript. Full documentation, additional examples, and citation information will be updated after publication.

## Citation

The manuscript is currently in revision. Citation information will be updated after publication.
The implementation is based on publicly available architectures, including Choo et al. (2025) and Srivastava et al. (2024), with modifications described in the Methods section of the manuscript.

References

Choo, M., Jung, S., Im, J., & Han, D. (2025). CARE-SST: Context-aware reconstruction diffusion model for sea surface temperature. ISPRS Journal of Photogrammetry and Remote Sensing, 220, 454–472.

Srivastava, P., Yang, R., Kerrigan, G., Dresdner, G., McGibbon, J., Bretherton, C. S., & Mandt, S. (2024). Precipitation downscaling with spatiotemporal video diffusion. Advances in Neural Information Processing Systems, 37, 56374–56400.

## Contact

For questions, please contact: Bokyung Son (sbkyung0@unist.ac.kr)




