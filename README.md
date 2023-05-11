# Nighttime Earth-light Analysis
Analyzing and Exploring Night VIIRS Satellite's nighttime Light data's connection with economic activity.


**Hypothesis:**

1. There is a significant positive correlation between the economic activity of a region and an increase in the night light brightness of human settlements.
---> A Significant positive correlation has indeed been found between the night light emissions and the regions economic activity
2. There is a dip in night light brightness during the coronavirus pandemic period in various countries.
---> No dip in light had been observed during the pandemic. While the exact reason cannot be said in a conclusive manner, one of the possible reasons could be the that the purchasing power of a consumer did not go down due the economic stimulus provided by the governments.

 

**Research questions:**

1. Can we identify rapidly growing economic clusters using night light data?
---> The rapidly growing economic clusters and cities have been visualized and observed in the analysis.
2. Can we identify activities in protected ecological areas?
---> There can not been significant correlation found between the forest cover loss rate and the night light trends.

**Initializing the project:**
After loading the project to your IDE. Refer to the instruction in the "initialize the dataset.ipynb" to get started.

**Notes and limitations**
1. The forest cover tiff file has not been trimmed to the national boundaries due to the processing power limitation of an average PC, however the all the required functions are made available in this program for you to test out if needed.
2. To create the sampling points dataset for other countries using QGIS refer to credits[7]. An alternative python method has also been documented in the sampling function in the python script
3. In view of GitHub's file size limitations an external link to india sampling file has been stored in the ./input data/sampling points/README.md

## Credits
The data has been obtained and refined from:
<ol>
    <li>The source data hass been imported form: https://eogdata.mines.edu/products/vnl/</li>
    <li>C. D. Elvidge, K. E. Baugh, M. Zhizhin, and F.-C. Hsu, “Why VIIRS data are superior to DMSP for mapping 
nighttime lights,” Asia-Pacific Advanced Network 35, vol. 35, p. 62, 2013.</li>
    <li>C. D. Elvidge, M. Zhizhin, T. Ghosh, F-C. Hsu, "Annual time series of global VIIRS nighttime lights derived 
from monthly averages: 2012 to 2019", Remote Sensing (In press)</li>
    <li>GDP (current US$) | Data from World bank</li>
    <li>India shape files: https://geodata.lib.utexas.edu/catalog/stanford-jm149wc6691</li>
    <li>germany shape files: https://www.diva-gis.org/gdata</li>
    <li> The creation of sampling coordinated from a country's national boundary shape files has been done using the instructions from 
https://github.com/thinkingmachines/ph-poverty-mapping/issues/27#issuecomment-570116153</li>
    <li> The tree cover datasets have been obtained from Source: Hansen/UMD/Google/USGS/NASA</li>
    <li>Hansen, M. C., P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland, A. Kommareddy, A. Egorov, L. Chini, C. O. Justice, and J. R. G. Townshend. 2013. “High-Resolution Global Maps of 21st-Century Forest Cover Change.” Science 342 (15 November): 850–53. Data available on-line from: https://glad.earthengine.app/view/global-forest-change.</li>
    <li>Assam oil well blowout: .wikipedia.org/wiki/2020_Assam_gas_and_oil_leak</li>
    <li>Delhi Smog od 2016 by business line https://www.thehindubusinessline.com/news/what-caused-the-great-delhi-smog-of-nov-2016/article30248782.ece</li>
</ol>
