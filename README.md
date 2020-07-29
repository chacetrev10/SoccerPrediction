## Soccer Prediction
With the limited amount of data provided to the public for Major League Soccer, I did my best to create a predictive model for outcomes of future games. The dataset that was used contained only scores of the past 2 seasons. With this I extracted several different features and compiled them in a dataframe for each team.
* AHG = Average goals a team scores when at home
* AAG = Average goals a team scores when away
* AHGA = Average goals a team concedes when at home
* AAGA = Average goals a team concedes when away

With a brief amount of research I used those averages for every team to compute the following:
* HAS = Home attack strength calculated by AHG/average goals scored at home in the MLS
* AAS = Away attack strength calculated by AAG/average goals scored away in the MLS
* HDS = Home defense strength calculated by AHGA/average goals conceded by teams at home in the MLS
* ADS = Away defense strength calculated by AAGA/average goals conceded by teams away in the MLS

All of these paramaters played a part in calcuating the following for every matchup
* HPG = The number of goals the home team is expected to score given its oppenent
* APG = The number of goals the away team is expected to score given its opponent

The equations to calculate the two were   

HPG = HAS * ADS * average goals scored at home in the MLS

APG = AAS * HDS * average goals scored away in the MLS

These two features were put into a support vector machine model where it was trained and supplied with data on future games which it produced predictions/probabilities for.
