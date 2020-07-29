import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




##read data
season = pd.read_csv('USA.csv')
cseason = season[['Home', 'Away', 'HG', 'AG', 'Res', 'Number']]
upcomingGameDF = pd.read_csv('nextGames.csv')
number = len(season.index)
averageDF = pd.DataFrame(columns=[ 'HT', 'AT','AHG', 'AAG', 'AHGA', 'AAGA', 'HAS', 'HDS','AAS'
                                  'ADS' , 'MLSHG', 'MLSAG','HPG', 'APG', 'Result'], index=[pd.RangeIndex(start=1, stop=number+1, step=1)])
currentDF = pd.DataFrame(columns=[ 'Team', 'AHG', 'AAG', 'AHGA', 'AAGA'], index=[pd.RangeIndex(start=1, stop= 27, step=1)])
averageDF['Result'] = season['Res']
nameAry = np.asanyarray(season.Home.unique())
print(nameAry)
currentDF['Team'] = nameAry



##methods
def mlsHomeGoalAverage():
    mlsHomeGoals = cseason['HG'].sum()
    return (mlsHomeGoals/number)

def mlsAwayGoalAverage():
    mlsAwayGoals = cseason['AG'].sum()
    return (mlsAwayGoals/number)


def averageHomeGoalsUpdate(name):
    i = 1
    teamDF = cseason.loc[cseason['Home'] == name]
    resultAry = np.asanyarray(teamDF[['Res']])
    ary1 = np.asanyarray(teamDF[['HG']])
    gameCount = np.asanyarray(teamDF[['Number']])



    while (i <= (teamDF.Home == name).sum()):

        games = i
        goals =0
        ary1 = np.asanyarray(teamDF[['HG']])
        current_game= ary1[:i].sum()
        goals += current_game
        ahg = goals / games
        index = gameCount[i-1]
        mlsUpdatingHomeGoals = np.asanyarray(cseason['HG']).astype(int)
        mlsUpdatingHomeGoals = mlsUpdatingHomeGoals[:index[0]].sum()
        mlsUpdatingHomeGoals = mlsUpdatingHomeGoals/ index[0]
        teamAS = ahg / mlsUpdatingHomeGoals
        result = resultAry[i-1]
        numberResult = 0
        if result == 'H':
            numberResult = 2
        elif result == 'D':
            numberResult = 1
        else:
            numberResult = 0

        averageDF.loc[index, 'Result'] = numberResult
        averageDF.loc[index , 'AHG'] = [ahg]
        averageDF.loc[index, 'HT'] = [name]
        averageDF.loc[index, 'HAS'] = [teamAS]
        averageDF.loc[index, 'MLSHG'] = [mlsUpdatingHomeGoals]
        i += 1

def averageHomeGoals(name):
    teamDF = cseason.loc[cseason['Home'] == name]
    goals = teamDF['HG'].sum()
    games = (teamDF.Home == name).sum()
    if games == 0:
        return 0
    else:
        ahg = goals / games

    return ahg


def averageAwayGoalsUpdate(name):
    i = 1
    teamDF = cseason.loc[cseason['Away'] == name]
    ary1 = np.asanyarray(teamDF[['AG']])
    gameCount = np.asanyarray(teamDF[['Number']])

    while (i <= (teamDF.Away == name).sum()):
        games = i
        goals =0

        ary1 = np.asanyarray(teamDF[['AG']])
        current_game= ary1[:i].sum()
        goals += current_game
        aag = goals / games
        index = gameCount[i-1]
        mlsUpdatingAwayGoals = np.asanyarray(cseason['AG']).astype(int)
        mlsUpdatingAwayGoals = mlsUpdatingAwayGoals[:index[0]].sum()
        mlsUpdatingAwayGoals = mlsUpdatingAwayGoals / index[0]
        teamAS = aag / mlsUpdatingAwayGoals
        averageDF.loc[index , 'AAG'] = [aag]
        averageDF.loc[index, 'AT'] = [name]
        averageDF.loc[index, 'AAS'] = [teamAS]
        averageDF.loc[index, 'MLSAG'] = [mlsUpdatingAwayGoals]
        i += 1



def averageAwayGoals(name):

    teamDF = cseason.loc[cseason['Away'] == name]
    goals = teamDF['AG'].sum()
    games = (teamDF.Away == name).sum()
    if games == 0:
        return 0
    else:
        aag = goals / games
    return aag




def averageHGagainstUpdate(name):
    i = 1
    teamDF = cseason.loc[cseason['Home'] == name]
    ary1 = np.asanyarray(teamDF[['AG']])
    gameCount = np.asanyarray(teamDF[['Number']])


    while (i <= (teamDF.Home == name).sum()):
        games = i
        goals =0

        ary1 = np.asanyarray(teamDF[['AG']])
        current_game= ary1[:i].sum()
        goals += current_game
        hga = goals / games
        index = gameCount[i-1]
        mlsUpdatingAwayGoals = np.asanyarray(cseason['AG']).astype(int)
        mlsUpdatingAwayGoals = mlsUpdatingAwayGoals[:index[0]].sum()
        mlsUpdatingAwayGoals = mlsUpdatingAwayGoals / index[0]
        teamDS = hga / mlsUpdatingAwayGoals
        averageDF.loc[index , 'AHGA'] = [hga]
        averageDF.loc[index, 'HDS'] = [teamDS]
        i += 1

def averageHGagainst(name):

    teamDF = cseason.loc[cseason['Home'] == name]
    goalsA = teamDF['AG'].sum()
    games = (teamDF.Home == name).sum()
    if games == 0:
        return 0
    else:
        HGA = goalsA / games
    return HGA


def averageAGagainstUpdate(name):
    i = 1
    teamDF = cseason.loc[cseason['Away'] == name]
    ary1 = np.asanyarray(teamDF[['HG']])
    gameCount = np.asanyarray(teamDF[['Number']])


    while (i <= (teamDF.Away == name).sum()):
        games = i
        goals =0

        ary1 = np.asanyarray(teamDF[['HG']])
        current_game= ary1[:i].sum()
        goals += current_game
        aga = goals / games
        index = gameCount[i-1]
        mlsUpdatingHomeGoals = np.asanyarray(cseason['HG']).astype(int)
        mlsUpdatingHomeGoals = mlsUpdatingHomeGoals[:index[0]].sum()
        mlsUpdatingHomeGoals = mlsUpdatingHomeGoals / index[0]
        teamDS = aga / mlsUpdatingHomeGoals
        averageDF.loc[index , 'AAGA'] = [aga]
        averageDF.loc[index, 'ADS'] = [teamDS]
        i += 1

def averageAGagainst(name):

    teamDF = cseason.loc[cseason['Away'] == name]
    goalsA = teamDF['HG'].sum()
    games = (teamDF.Away == name).sum()
    if games == 0:
        return 0
    else:
        AGA = goalsA / games
    return AGA

def homeTeamAttackStrength(name):
    homeAttack = averageHomeGoals(name)/mlsHomeGoalAverage()
    return homeAttack

def awayTeamAttackStrength(name):
    awayAttack = averageAwayGoals(name)/mlsAwayGoalAverage()
    return awayAttack

def awayTeamDefenseStrength(name):
    awayDefense = averageAGagainst(name)/mlsHomeGoalAverage()
    return awayDefense

def homeTeamDefenseStrength(name):
    homeDefense = averageHGagainst(name)/mlsAwayGoalAverage()
    return homeDefense

##call method
name = '1'
teamCount = 0
currentTeamStat = 0



##updating stats
while (teamCount < len(nameAry)):
    name = nameAry[teamCount]
    averageHomeGoalsUpdate(name)
    averageAwayGoalsUpdate(name)
    averageHGagainstUpdate(name)
    averageAGagainstUpdate(name)


    teamCount +=1

predictingGoalCount = 1
while(predictingGoalCount <= len(season)):
    averageDF.loc[predictingGoalCount, 'HPG'] = [
        (averageDF.loc[predictingGoalCount, 'HAS']) * (averageDF.loc[predictingGoalCount, 'ADS']) * (averageDF.loc[
            predictingGoalCount, 'MLSHG'])]
    averageDF.loc[predictingGoalCount, 'APG'] = [
        (averageDF.loc[predictingGoalCount, 'AAS']) * (averageDF.loc[predictingGoalCount, 'HDS']) * (averageDF.loc[
            predictingGoalCount, 'MLSAG'])]

    predictingGoalCount += 1



##current stats
while (currentTeamStat < len(nameAry)):
    name = nameAry[currentTeamStat]
    currentDF.loc[currentTeamStat + 1, 'AHG'] = [averageHomeGoals(name)]
    currentDF.loc[currentTeamStat + 1, 'AAG'] = [ averageAwayGoals(name)]
    currentDF.loc[currentTeamStat + 1, 'AAGA'] = [averageAGagainst(name)]
    currentDF.loc[currentTeamStat + 1, 'AHGA'] = [averageHGagainst(name)]

    currentTeamStat +=1


##upcoming game stats
currentTeamStat = 0
print(len(upcomingGameDF['HT']))
while (currentTeamStat < len(upcomingGameDF['HT'])):

    home = upcomingGameDF.loc[currentTeamStat , 'HT']
    away = upcomingGameDF.loc[currentTeamStat , 'AT']

    upcomingGameDF.loc[currentTeamStat , 'AHG'] = [averageHomeGoals(home)]
    upcomingGameDF.loc[currentTeamStat , 'AAG'] = [averageAwayGoals(away)]
    upcomingGameDF.loc[currentTeamStat , 'AAGA'] = [averageAGagainst(away)]
    upcomingGameDF.loc[currentTeamStat , 'AHGA'] = [averageHGagainst(home)]
    upcomingGameDF.loc[currentTeamStat , 'HAS'] = [homeTeamAttackStrength(home)]
    upcomingGameDF.loc[currentTeamStat, 'HDS'] = [homeTeamDefenseStrength(home)]
    upcomingGameDF.loc[currentTeamStat , 'AAS'] = [awayTeamAttackStrength(away)]
    upcomingGameDF.loc[currentTeamStat ,'ADS'] = [awayTeamDefenseStrength(away)]
    upcomingGameDF.loc[currentTeamStat, 'ADS'] = [awayTeamDefenseStrength(away)]
    upcomingGameDF.loc[currentTeamStat, 'HPG'] = [homeTeamAttackStrength(home)*awayTeamDefenseStrength(away)*mlsHomeGoalAverage()]
    upcomingGameDF.loc[currentTeamStat, 'APG'] = [awayTeamAttackStrength(away) * homeTeamDefenseStrength(home) * mlsAwayGoalAverage()]

    currentTeamStat += 1



averageDF.to_csv('averageMLS.csv', sep=',')
currentDF.to_csv('currentMLS.csv', sep=',')



##plot
averageDF['AHG'] = averageDF['AHG'].astype(float)
averageDF['AAG'] = averageDF['AAG'].astype(float)
ax = averageDF[averageDF['Result'] == 2][1:100].plot(kind='scatter', x='AHG', y='AAG', color='DarkBlue', label='Home win')
averageDF[averageDF['Result'] == 0][1:100].plot(kind='scatter', x='AHG', y='AAG', color='Yellow', label='Home Loss', ax=ax)
averageDF[averageDF['Result'] == 1][1:100].plot(kind='scatter', x='AHG', y='AAG', color='Red', label='Home Tie', ax=ax)
plt.show()



##split data
featureDF = averageDF[['HPG','APG']]
X = np.asanyarray(featureDF)

y = np.asarray(averageDF['Result'])
y [0:5]


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=2)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

X_train = X_train.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_test = y_test.astype(float)

##svm
from sklearn import svm
clf = svm.SVC(kernel = 'rbf', gamma = 'scale', probability= True)
clf.fit(X_train, y_train)



##predicting upcoming games
nextGamesFeatures =upcomingGameDF[['HPG','APG']]

gameProb = clf.predict_proba(nextGamesFeatures)
appendDf = pd.DataFrame(gameProb, columns = ['Away Win', 'Tie', 'Home Win'])
gamePred = clf.predict(nextGamesFeatures)
upcomingGameDF['Pred'] = gamePred
upcomingGameDF = pd.concat([upcomingGameDF, appendDf], axis = 1)
upcomingGameDF.to_csv('nextGames.csv')

yhat = clf.predict(X_test)

# get the accuracy
print(yhat)
print(y_test)
print (accuracy_score(y_test, yhat))

