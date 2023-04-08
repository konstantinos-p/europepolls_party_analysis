<h2 align="center">Analysis utils for polling data.</h2>

These functions can be used to analyse polling data. In particular, one can compute trend lines with Gaussian Processes 
compute the bias of pollsters and poll commissioners, and finally debias available polls.

<h2> :red_book: Usage.</h2>
The usage of each function is explained in detail analysis_utils.py.

<h2> :blue_book: Format.</h2>
The functions in analysis_utils.py expect as input pandas.dataframe objects with the following structure.

```

| Date                 | Polling Firm | Commissioner | Sample Size | Named Parties | Others     |
|----------------------|--------------|--------------|-------------|---------------|------------|
| :obj:pandas.datetime | :obj:str     | :obj:str     | :obj:float  | :obj:float    | :obj:float |

```
A brief explanation of each entry is given below.

- **Date**: The date when the poll was conducted. In some cases both a start and an end date is reported in the raw data,
corresponding to the start and the end of the polling. In this case we report the end date.
- **Polling Firm**: The name of the firm conducting the poll.
- **Commissioner**: The name of the public or private entity which commissioned the poll.
- **Sample Size**: The size of the sample of the poll.
- **Named Parties**: All parties with high enough polling numbers such that they are included by name in the poll.
- **Others**: This entry includes the polling total of all parties polling too low to be included by name in the results.

Undecided voters are excluded when applicable. The results for each party are then normalized to reflect the polling
percentage over all valid votes.




<h2> :envelope: Contact Information </h2>
You can contact me at any of my social network profiles:

- :briefcase: Linkedin: https://www.linkedin.com/in/konstantinos-pitas-lts2-epfl/
- :octocat: Github: https://github.com/konstantinos-p

Or via email at cwstas2007@hotmail.com

<h2> :warning: Disclaimer </h2>
This Python package has been made for research purposes.

