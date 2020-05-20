{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analyzing the Board Game _TEG_\n",
    "\n",
    "Falta reemplazar las instancias de Risk por TEG\n",
    "\n",
    "This notebook shows you how to use the probability tools from this site to analyze the odds in [_Risk_](https://en.wikipedia.org/wiki/Risk_(game%29), so you can make better decisions and hopefully improve your chances of winning.\n",
    "\n",
    "If you're not familiar with _Risk_, you can [find an overview of how to play the game here](http://www.ultraboardgames.com/risk/game-rules.php)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from collections import deque, defaultdict\n",
    "import itertools as it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "pd.options.display.float_format = '{:.3f}'.format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib as mpl\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()\n",
    "sns.set_context('notebook')\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This notebook uses the probability modules from the `pracpred` package developed on this site. You can [find more information about the package here](https://github.com/practicallypredictable/pracpred) and [here](https://pypi.python.org/pypi/pracpred/0.1.2)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pracpred.prob import Prob, ProbDist"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "_Risk_ uses up to 5 six-sided dice to determine the outcome of attacks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 275,
   "metadata": {},
   "outputs": [],
   "source": [
    "d6 = ProbDist(range(1,7))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ProbDist({1: Prob(1, 6), 2: Prob(1, 6), 3: Prob(1, 6), 4: Prob(1, 6), 5: Prob(1, 6), 6: Prob(1, 6)})"
      ]
     },
     "execution_count": 276,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d6"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Attacks in _Risk_\n",
    "\n",
    "We are going to use the term _attack_ to mean a clash between armies settled by a dice roll. In _Risk_, any particular attack can have up to 3 attacking armies and up to 2 defending armies. There are only 6 valid attacks."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 277,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Attacks are represented as a tuple of (attacker armies, defender armies)\n",
    "# Attacks do not include the attacker army required to remain in the territory from which the attack is launched\n",
    "VALID_ATTACKS = [\n",
    "    (1, 1),\n",
    "    (1, 2),\n",
    "    (1, 3),\n",
    "    (2, 1),\n",
    "    (2, 2),\n",
    "    (2, 3),\n",
    "    (3, 1),\n",
    "    (3, 2),\n",
    "    (3, 3),\n",
    "]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are going to use the term _battle_ to mean the larger conflict between two players over a particular territory. For example, if the attacker has 20 armies in China and wishes to attack the defender's 15 armies in Siberia, we call that a 19-on-15 battle. (In _Risk_, the attacker is required to keep one army in the territory from which he or she is attacking.)\n",
    "\n",
    "Within this larger battle, each roll of the dice is an attack, which is at most 3-on-2. It's important to be clear about the difference between battles and attacks, since the attacker can halt the battle at any point, and potentially do something completely different.\n",
    "\n",
    "We will begin defining some Python functions to help us represent the board game."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 278,
   "metadata": {},
   "outputs": [],
   "source": [
    "def attackers(attack):\n",
    "    \"\"\"Number of attackers in an attack.\"\"\"\n",
    "    return attack[0]\n",
    "\n",
    "def defenders(attack):\n",
    "    \"\"\"Number of defenders in an attack.\"\"\"\n",
    "    return attack[1]\n",
    "\n",
    "def total_armies(attack):\n",
    "    \"\"\"Total number of armies on both sides involved in an attck.\"\"\"\n",
    "    return attackers(attack)+defenders(attack)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Dice Probabilities\n",
    "\n",
    "We can compute all the possible dice rolls in advance, since there are only at most 5 dice rolled."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 279,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_roll_probs(attacks):\n",
    "    \"\"\"Probability distribution of rolls in attacks, by total number of dice rolled.\"\"\"\n",
    "    return {total_armies(attack): d6.repeated(total_armies(attack), product=True) for attack in attacks}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 280,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roll_probs = get_roll_probs(VALID_ATTACKS)\n",
    "len(roll_probs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 281,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys([2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 281,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roll_probs.keys()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Every attack has at least 2 dice (one for the attacker and one for the defender). Notice also that we are not distinguishing between attacker and defender dice yet. This means that the 2-on-2 attack and the 3-on-1 attack will both use the roll probabilities for 4 dice, for example.\n",
    "\n",
    "Let's look at the number of outcomes in the distribution for each set of dice."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 282,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2 36\n",
      "3 216\n",
      "4 1296\n",
      "5 7776\n",
      "6 46656\n"
     ]
    }
   ],
   "source": [
    "for armies in roll_probs:\n",
    "    print(armies, len(roll_probs[armies]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In our [previous post on dice rolls](http://practicallypredictable.com/2017/12/04/probability-distributions-dice-rolls/), we looked at the distribution of the sum of six-sided dice. For _Risk_, we need to generate the full [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of possible dice outcomes, because we are comparing the attacker and defender dice individually. That's why the number of outcomes for each set of dice is $6^n$, where $n$ is the number of dice rolled.\n",
    "\n",
    "### Attacker versus Defender Rolls\n",
    "\n",
    "Now we need to separately keep track of the attacker dice and the defender dice. We also need to sort each set of dice so we can determine the outcome of an attack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 283,
   "metadata": {},
   "outputs": [],
   "source": [
    "def attacker_roll(attack, roll):\n",
    "    \"\"\"Dice rolled by attacker, sorted highest to lowest.\"\"\"\n",
    "    return sorted(roll[:attackers(attack)], reverse=True)\n",
    "\n",
    "def defender_roll(attack, roll):\n",
    "    \"\"\"Dice rolled by defender, sorted highest to lowest.\"\"\"\n",
    "    return sorted(roll[-defenders(attack):], reverse=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We'll discuss in a little bit how to figure out how many dice to roll for a given attack.\n",
    "\n",
    "Let's simulate a 3-on-2 attack to see how these functions work."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 284,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example 3-on-2 attack\n",
    "attack = (3, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 285,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4, 3, 5, 4, 3, 6)"
      ]
     },
     "execution_count": 285,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roll = roll_probs[total_armies(attack)].choice()\n",
    "roll"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 286,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[5, 4, 3]"
      ]
     },
     "execution_count": 286,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "attacker_roll(attack, roll)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 287,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[6, 4, 3]"
      ]
     },
     "execution_count": 287,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "defender_roll(attack, roll)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Armies Lost\n",
    "\n",
    "Now we can figure out how many armies each side loses in the attack. We compare the highest die roll by the attacker to the highest die roll by the defender. The attacker loses an army if his or her roll is less than or equal to the defender's roll. Otherwise, the defender loses an army.\n",
    "\n",
    "If the attacker is rolling more than one die, and the defender is rolling two dice, we compare the attacker's second-highest die roll with the defender's lower die roll. The attacker loses an army if his or her second die roll is less than or equal to the defender's lower die roll. Otherwise, the defender loses an army.\n",
    "\n",
    "In one attack, the most armies either side can lose is equal to the number of dice the defender rolls.\n",
    "\n",
    "This function will determine how many armies each side loses in a particular attack given the dice rolls."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 288,
   "metadata": {},
   "outputs": [],
   "source": [
    "# EDITAR PARA CONSIDERAR TERCER DADO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
    "def losses_from_roll(att_roll, def_roll):\n",
    "    \"\"\"Armies lost by (attacker, defender) based on dice roll.\"\"\"\n",
    "    if att_roll[0] > def_roll[0]:\n",
    "        att_loses = 0\n",
    "        def_loses = 1\n",
    "    else:\n",
    "        att_loses = 1\n",
    "        def_loses = 0\n",
    "    if len(def_roll) > 1 and len(att_roll) > 1:\n",
    "        if att_roll[1] > def_roll[1]:\n",
    "            def_loses += 1\n",
    "        else:\n",
    "            att_loses += 1\n",
    "    if len(def_roll) > 2 and len(att_roll) > 2:\n",
    "        if att_roll[2] > def_roll[2]:\n",
    "            def_loses += 1\n",
    "        else:\n",
    "            att_loses += 1\n",
    "    return (att_loses, def_loses)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 289,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 0)"
      ]
     },
     "execution_count": 289,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "losses_from_roll(attacker_roll(attack, roll), defender_roll(attack, roll))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above function can tell us how many armies each side loses for a given attacker roll and defender roll. To use the function, though, we first need to separate out the attacker dice from the defender dice.\n",
    "\n",
    "### Armies Lost for Every Possible Attack\n",
    "\n",
    "What we really want is for each possible type of attack to have it's own function, which will compute the armies lost for any particular dice roll. Here's a neat trick using [nested functions in Python](https://realpython.com/blog/python/inner-functions-what-are-they-good-for/) to do that."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 290,
   "metadata": {},
   "outputs": [],
   "source": [
    "def armies_lost(attack):\n",
    "    \"\"\"Create a function to calculate armies lost by (attacker, defender) for a given attack.\"\"\"\n",
    "    def inner_func(roll):\n",
    "        att_roll = attacker_roll(attack, roll)\n",
    "        def_roll = defender_roll(attack, roll)\n",
    "        return losses_from_roll(att_roll, def_roll)\n",
    "    return inner_func"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 291,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "function"
      ]
     },
     "execution_count": 291,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(armies_lost(attack))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So this function returned a function? Yes, in Python you can do that. The expression `armies_lost(attack)` looks like it should return a value of some sort, but in this case it returns a function.\n",
    "\n",
    "Remember that the example attack we were playing with above is 3-on-2. The point of the nested function is that the `attack` parameter is known to the `inner_func` function, and gets hidden inside it. The inner function will always \"remember\" that it was initially called with the 3-on-2 attack parameter. We can call the returned function now with a particular dice roll, and the function will assume a 3-on-2 attack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 292,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 0)"
      ]
     },
     "execution_count": 292,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "armies_lost(attack)(roll)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we can build a function like this for every possible attack. Think of the `armies_lost()` function as a \"factory\" that builds and returns other functions, based on what type of attack we tell it to build."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_attack_losses(attacks):\n",
    "    \"\"\"Functions to calculate armies lost by attacker and defender for each type of attack.\"\"\"\n",
    "    return {attack: armies_lost(attack) for attack in attacks}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{(1, 1): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (1, 2): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (1, 3): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (2, 1): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (2, 2): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (2, 3): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (3, 1): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (3, 2): <function __main__.armies_lost.<locals>.inner_func(roll)>,\n",
       " (3, 3): <function __main__.armies_lost.<locals>.inner_func(roll)>}"
      ]
     },
     "execution_count": 294,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "attack_losses = get_attack_losses(VALID_ATTACKS)\n",
    "attack_losses"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's take a moment to review what's going on here. We now have 6 functions, one for each possible attack type. Given the type of attack, we can look up the right function to use and call it on a particular dice roll."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 0)"
      ]
     },
     "execution_count": 295,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "attack_losses[attack](roll)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's test our function on 10 random 3-on-2 attacks to get a feel for how it works."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Att = [6, 6, 2], Def = [5, 4, 1], losses = (0, 3)\n",
      "Att = [5, 5, 5], Def = [1, 1, 1], losses = (0, 3)\n",
      "Att = [6, 5, 4], Def = [4, 3, 2], losses = (0, 3)\n",
      "Att = [6, 2, 2], Def = [4, 3, 1], losses = (1, 2)\n",
      "Att = [6, 6, 1], Def = [3, 2, 2], losses = (1, 2)\n",
      "Att = [6, 4, 3], Def = [6, 2, 1], losses = (1, 2)\n",
      "Att = [6, 6, 2], Def = [4, 4, 2], losses = (1, 2)\n",
      "Att = [6, 6, 2], Def = [6, 3, 1], losses = (1, 2)\n",
      "Att = [6, 2, 1], Def = [4, 4, 2], losses = (2, 1)\n",
      "Att = [5, 4, 2], Def = [1, 1, 1], losses = (0, 3)\n"
     ]
    }
   ],
   "source": [
    "for roll in roll_probs[total_armies(attack)].sample(10):\n",
    "    print('Att = {att_roll}, Def = {def_roll}, losses = {losses}'.format(\n",
    "        att_roll=attacker_roll(attack, roll),\n",
    "        def_roll=defender_roll(attack, roll),\n",
    "        losses=attack_losses[attack](roll))\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Probability of Attack Outcomes\n",
    "\n",
    "Now we can start analyzing probabilities of attack outcomes. For each type of possible attack, we need to get all the possible dice rolls for that type, and then group the outcomes by how many armies are lost by each side.\n",
    "\n",
    "This is reason I created a different army loss function for each type of attack. It makes it easy to do all the grouping in our probability framework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ProbDist({(0, 3): Prob(535, 3888), (1, 2): Prob(371, 1728), (2, 1): Prob(343, 1296), (3, 0): Prob(5957, 15552)})"
      ]
     },
     "execution_count": 297,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "roll_probs[total_armies(attack)].groupby(attack_losses[attack])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Take a moment to unpack the above line. For the given attack type (in this example, 3-on-2), get the number of dice to roll (in this case, 5), and get all of the 7776 possible rolls for 5 dice. Then group those rolls by how many armies are lost in each roll outcome.\n",
    "\n",
    "That's a lot going on in one line of Python. But it shows the power and generality of the probabilty modeling framework.\n",
    "\n",
    "Now we can generate the distribution of armies lost for every possible attack, not just 3-on-2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_loss_probs(attacks, roll_probs=None, attack_losses=None):\n",
    "    \"\"\"Distribution of armies lost by (attacker, defender) in an attack, by attack.\"\"\"\n",
    "    if not roll_probs:\n",
    "        roll_probs = get_roll_probs(attacks)\n",
    "    if not attack_losses:\n",
    "        attack_losses = get_attack_losses(attacks)\n",
    "    return {attack: roll_probs[total_armies(attack)].groupby(attack_losses[attack]) for attack in attacks}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9"
      ]
     },
     "execution_count": 299,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# TIENE QUE DAR 9 ?!!!!!!\n",
    "loss_probs = get_loss_probs(VALID_ATTACKS, roll_probs, attack_losses)\n",
    "len(loss_probs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1, 1) {(0, 1): 5/12, (1, 0): 7/12}\n",
      "(1, 2) {(0, 1): 55/216, (1, 0): 161/216}\n",
      "(1, 3) {(0, 1): 25/144, (1, 0): 119/144}\n",
      "(2, 1) {(0, 1): 125/216, (1, 0): 91/216}\n",
      "(2, 2) {(0, 2): 295/1296, (1, 1): 35/108, (2, 0): 581/1296}\n",
      "(2, 3) {(0, 2): 979/7776, (1, 1): 1981/7776, (2, 0): 301/486}\n",
      "(3, 1) {(0, 1): 95/144, (1, 0): 49/144}\n",
      "(3, 2) {(0, 2): 1445/3888, (1, 1): 2611/7776, (2, 0): 2275/7776}\n",
      "(3, 3) {(0, 3): 535/3888, (1, 2): 371/1728, (2, 1): 343/1296, (3, 0): 5957/15552}\n"
     ]
    }
   ],
   "source": [
    "for attack in loss_probs:\n",
    "    print(attack, loss_probs[attack])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "That's it. Those are the possible outcomes in terms of armies lost in any given _Risk_ attack, along with the probability of that outcome happening.\n",
    "\n",
    "### Battle Outcomes\n",
    "\n",
    "It's great to analyze the armies lost in a given attack. But, what we ultimately are about is probability of winning a battle. Let's look at how to do that now.\n",
    "\n",
    "For the attacker to win a battle, the defender's armies must be reduced to zero in the attacked territory. For the defender to win the battle, either the attacker's armies are reduced to zero (not including the one army that's required to remain behind), or the attacker calls of the attack.\n",
    "\n",
    "#### A Simplified Example\n",
    "\n",
    "Let's compute \"by hand\" the probability that the attacker wins a 2-on-1 battle. Later we'll figure out how to automate this for any possible battle.\n",
    "\n",
    "In a 2-on-1 battle, the first attack is also 2-on-1. The possible outcomes are: the attacker loses an army, or the defender loses an army. If the defender loses an army, the battle is over.\n",
    "\n",
    "The probability this happens is:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(125, 216)"
      ]
     },
     "execution_count": 301,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W1 = loss_probs[(2, 1)][(0, 1)]\n",
    "pA_W1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On the other hand, the attacker loses an army with probability:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(91, 216)"
      ]
     },
     "execution_count": 302,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_L1 = 1-pA_W1\n",
    "pA_L1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's suppose the attacker is really aggressive and decides to continue the battle. This next attack would then be 1-on-1. The attacker wins this round with probability:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(5, 12)"
      ]
     },
     "execution_count": 303,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W2 = loss_probs[(1, 1)][(0, 1)]\n",
    "pA_W2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On the other hand, the defender prevails with probability:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(7, 12)"
      ]
     },
     "execution_count": 304,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_L2 = 1-pA_W2\n",
    "pA_L2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The defender has the advantage in the 1-on-1 attack because the defender wins ties.\n",
    "\n",
    "We assume that each dice roll is independent as usual. That means that we can use the [>>>multiplication rule for probabilities]() to combine events. In this case, the combined event we care about is that the attacker loses the first attack but wins the second attack. This probability is:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(455, 2592)"
      ]
     },
     "execution_count": 305,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_L1W2 = pA_L1*pA_W2\n",
    "pA_L1W2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The overall probability the attacker wins the battle is the probability of winning in the first round, plus the probability of winning in the second round. The overall probability the attacker wins the 2-on-1 battle is:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(1955, 2592)"
      ]
     },
     "execution_count": 306,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W = pA_W1 + pA_L1W2\n",
    "pA_W"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Similarly, the probability the attacker loses is the probability of losing in the first round times the probability of losing in the second round. This probability is:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(637, 2592)"
      ]
     },
     "execution_count": 307,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_L = pA_L1*pA_L2\n",
    "pA_L"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You may wonder, why do we **add** the probabilities for winning, but **multiply** the probabilities for losing?\n",
    "\n",
    "Great question. It's because the attacker **either** wins in the first round, **or** she wins in the second round. They both cannot be true. In probability theory, these are called [mutually exclusive events](https://en.wikipedia.org/wiki/Mutual_exclusivity). You can add probabilities for mutually exclusive events because they don't overlap, if you want to know the probability that either one or the other event happens.\n",
    "\n",
    "On the other hand, for the attacker to lose under our assumptions, she must lose the first round **and** the second round. Since these are independent events, we multiply probabilities to determine the probability that they both happen.\n",
    "\n",
    "Of course, the probability that the attacker either wins or loses is one:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(1, 1)"
      ]
     },
     "execution_count": 308,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W + pA_L"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Some Assumptions\n",
    "\n",
    "We made a few assumptions in the 2-on-1 battle example. First, we assumed that the first attack was 2-on-1. It might seem obvious that the attacker should use all her armies, but _Risk_ gives the players some choices on how many dice to roll in each attack.\n",
    "\n",
    "It's an interesting question whether the attacker and defender should always roll the maximum number of dice allowed. We will look at that question in a future post. For now, let's make the common choice of always rolling the maximum number of dice.\n",
    "\n",
    "We also assumed that the attacker continues to attack until losing the last possible army. We'll examine in a future post whether this is the best choice to make.\n",
    "\n",
    "### Modeling Battles in Python\n",
    "\n",
    "Let's define some more Python functions to help us represent any possible _Risk_ battle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [],
   "source": [
    "def attack_with(armies):\n",
    "    \"\"\"Attack with maximum number of armies allowed.\"\"\"\n",
    "    return min(3, attackers(armies))\n",
    "\n",
    "def defend_with(armies):\n",
    "    \"\"\"Defend with maximum number of armies allowed.\"\"\"\n",
    "    return min(3, defenders(armies))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example battle starting armies\n",
    "start = (3, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 311,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "attack_with(start)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 312,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "defend_with(start)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [],
   "source": [
    "def armies_left(armies, losses):\n",
    "    \"\"\"Armies remaining after losses.\"\"\"\n",
    "    return (attackers(armies)-attackers(losses), defenders(armies)-defenders(losses))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2, 3)"
      ]
     },
     "execution_count": 314,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "armies_left(start, (1, 0))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### More on Attack Outcomes\n",
    "\n",
    "Now we want to focus on generating all the attack outcomes and their probabilities. We will also put each attack together into the larger context of the battle.\n",
    "\n",
    "Recall that we already have all the probabilities for how many armies each side can lose for a given attack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ProbDist({(0, 3): Prob(535, 3888), (1, 2): Prob(371, 1728), (2, 1): Prob(343, 1296), (3, 0): Prob(5957, 15552)})"
      ]
     },
     "execution_count": 315,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
    "loss_probs[start]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we want to compute the probabilities for how many armies each side has left after a given attack."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "def attack_outcomes(armies):\n",
    "    \"\"\"Distribution of remaining (attacker, defender) armies after an attack.\"\"\"\n",
    "    attack = (attack_with(armies), defend_with(armies))\n",
    "    return {armies_left(armies, losses): loss_probs[attack][losses] for losses in loss_probs[attack]}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{(0, 3): Prob(5957, 15552),\n",
       " (1, 2): Prob(343, 1296),\n",
       " (2, 1): Prob(371, 1728),\n",
       " (3, 0): Prob(535, 3888)}"
      ]
     },
     "execution_count": 317,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "attack_outcomes(start)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Deciding When a Battle is Finished\n",
    "\n",
    "Under our assumptions, a battle is over when either the attacker or defender has zero armies left."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [],
   "source": [
    "def finished(attack):\n",
    "    \"\"\"True if either attacker or defender has lost battle.\"\"\"\n",
    "    return attackers(attack) == 0 or defenders(attack) == 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [],
   "source": [
    "def attacker_wins(outcome):\n",
    "    \"\"\"True if attacker won battle.\"\"\"\n",
    "    return attackers(outcome) > 0 and defenders(outcome) == 0\n",
    "\n",
    "def defender_wins(outcome):\n",
    "    \"\"\"True if defender won battle.\"\"\"\n",
    "    return attackers(outcome) == 0 and defenders(outcome) > 0"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Markov Chains\n",
    "\n",
    "The hard part about modeling battles in _Risk_ is keeping track of all the possible ways a battle could go. Things were relatively simple for the 2-on-1 example, but for a 10-on-6 attack there are many more possibilities. We will look at exactly how many possibilities there are for various _Risk_ battles in a future post.\n",
    "\n",
    "For now, I want to show you one way to solve this counting problem in Python, so you can see some useful results. This code may be a little hard to understand, but it's not that  different from how you would solve this  problem by hand if you had to. The framework used to model a general battle is called a [_Markov chain_](https://en.wikipedia.org/wiki/Markov_chain). Markov chains are named after the important Russian mathematician [Andrey Markov](https://en.wikipedia.org/wiki/Andrey_Markov). We will use Markov chains a lot on this site to analyze sports. This example using _Risk_ is meant as a simple introduction.\n",
    "\n",
    "The basic idea is to start at the beginning of the battle. Since we know how many armies each side has, we know how many dice will be rolled in the first attack. We also can figure out the possible outcomes of that attack, and the distribution of possible armies each side will have left after the first attack.\n",
    "\n",
    "#### State Space\n",
    "\n",
    "Consider the _Risk_ battle as a random process. We can think of the number of armies each side has as the _state_ of the battle at any point in time. A [_state space_](https://en.wikipedia.org/wiki/State_space) is a set of values which a process can be in at any particular time. The state space has to tell us everything we would need to compute the probabilities for the next possible value of the process. The number of armies satisfies this requirement in _Risk_. If we know the number of armies, we know the number of dice to roll at each step, so we can figure out all the probabilities at each step.\n",
    "\n",
    "#### Markov Property\n",
    "\n",
    "Here's the important insight about _Risk_. The only thing that matters is the current state (i.e., how many armies each side has for the next attack). If the next attack is 3-on-2, it doesn't matter if the battle started 10-on-6 or 50-on-39. We may feel worse as the attacker if we started 50-on-39, but the probabilities of the 3-on-2 attack outcomes are the same either way.\n",
    "\n",
    "This means that a _Risk_ battle satisfies the [_Markov property_](https://en.wikipedia.org/wiki/Markov_property). A random process has the Markov property if the future outcomes of the process only depend on the _current state_ of the process. Another way of saying this is that a process has the Markov property if it doesn't remember how it got to the current state. This is true of _Risk_.\n",
    "\n",
    "#### Counting Battle States\n",
    "\n",
    "You may wonder why this matters. It matters because we don't need to keep track of all the possible paths in a battle if the battle is a Markov process. We only need to keep track of the probability of being in each state. We can essentially forget all the prior information about how we got into a particular state.\n",
    "\n",
    "This significantly lowers the difficulty in counting, as you'll see. There are much fewer states than the number of possible paths through a battle.\n",
    "\n",
    "The initial state of the battle is just the number of armies each side has to start. Since the number of armies can never go up, we only have to represent the states that have armies less than or equal to the initial number, down to at worst zero armies for one side or the other. There are a finite number of possible states for any _Risk_ battle.\n",
    "\n",
    "If we ever get to a state where one side has zero armies, the battle is over and the side with some armies remaining is the winner.\n",
    "\n",
    "#### Accumulating the Attack Probabilities\n",
    "\n",
    "In _Risk_, each attack is independent of the prior attacks in a battle. This is true because the dice rolls are independent.\n",
    "\n",
    "As we saw in the [prior notebook on joint probabilities of independent events](https://github.com/practicallypredictable/posts/blob/master/notebooks/probability-part3-coin_flips-mult_probabilities.ipynb), you multiply the probabilities of two independent events to get the probability of the joint event that both occur.\n",
    "\n",
    "In general, let's define the probability $P(S_{a,d})$ that the battle is ever in state $S_{a,d}$, where _a_ is the number of attacker armies and _d_ is the number of defender armies. Suppose we start the battle with 10 attacker and 6 defender armies. This state is $S_{10,6}$. We know there are three possible attack outcomes: the attacker loses 2 armies, the defender loses 2 armies, and each loses an army. Therefore, from the state $S_{10,6}$, the next possible states are $S_{8,6}$, $S_{9,5}$ and $S_{10,4}$. \n",
    "\n",
    "How does $P(S_{8,6})$ depend on $P(S_{10,6})$?\n",
    "\n",
    "Think of $S_{8,6}$ being a joint probability based on two outcomes. The first outcome is that we were in $S_{10,6}$ on the prior step. The second outcome is that the attacker lost 2 armies in the 3-on-2 dice roll. Let's call this probability $P(A_{-2,3v2})$. These outcomes are independent, so we multiply the probabilities.\n",
    "\n",
    "So is the following equation true?\n",
    "$$P(S_{8,6}) = P(S_{10,6}) \\times P(A_{-2,3v2})$$\n",
    "\n",
    "In this case, yes, because there is only way one for the battle to get into the state $S_{8,6}$. If we were looking at the state $S_{7,5}$, we would have to be more careful. There are two ways we could get to $S_{7,5}$. The first is from $S_{8,6}$, after which the attacker and defender each lose an army. The other is from $S_{9,5}$, after which the attacker loses 2 armies.\n",
    "\n",
    "Remember earlier in this post, we talked about mutually exclusive events. You can add probabilities of mutually exclusive events because the events don't overlap. In this example _Risk_ battle, $S_{8,6}$ and $S_{9,5}$ are mutually exclusive events. There is no way the battle can ever have been in one state if it was ever in the other state. Once you know the outcome of the first attack, you must be in one, and only one, of $S_{8,6}$, $S_{9,5}$ or $S_{10,4}$.\n",
    "\n",
    "Now we have enough information to calculate $P(S_{7,5})$.\n",
    "$$P(S_{7,5}) = P(S_{8,6}) \\times P(A_{-1,3v2}) + P(S_{9,5}) \\times P(A_{-2,3v2})$$\n",
    "\n",
    "All this equation is saying is that the probability of being in any state equals the probability of being in a possible prior state, times the probability of going from that prior state to the current state, summed over all possible prior states.\n",
    "\n",
    "This is true for any state. The probability of going from one state to the other is just the various attack outcome probabilities, based upon the number of dice rolled after the prior state. Remember that you can figure out how many dice to roll in each attack, based upon the number of armies each side has in the prior state.\n",
    "\n",
    "The results we care about are the probabilities of ending up in a state where one side has zero armies and the other side has a positive number of armies. Once we know the probabilities in each of these ending states (called _terminal states_), we can figure out the probabilities that each side won or lost the battle.\n",
    "\n",
    "#### The Algorithm\n",
    "\n",
    "The only tricky part of all this is keeping track of the possible states and the probabilities of being in each.\n",
    "\n",
    "Here's a relatively simple approach to solve this problem. First, we set the probability of the initial state to 1. The probabilities of all other possible states start at zero.\n",
    "\n",
    "Next, we look at all possible outcomes from the starting attack. We save these outcomes in a data structure which computer scientists call a [queue](https://en.wikipedia.org/wiki/Queue_(abstract_data_type)). In Python, we can represent a queue with the `deque` class from the Python standard library.\n",
    "\n",
    "We multiply the initial probability (which is 1) times each attack outcome probability, and store the results of each multiplication. To store the probabilities, we use a Python `defaultdict` structure with the state as the key. We use `defaultdict` so that the probability defaults to zero for any state we haven't looked at already.\n",
    "\n",
    "Once we've examined the possible outcomes from the initial state, we've \"used up\" the probability (initially 1) for that state. We can't end the battle at the starting point. So we erase the starting point from the outcomes, and look at whatever attack is next in the queue. This new attack will have its own possible outcomes, which are also saved in the queue if they aren't already there.\n",
    "\n",
    "If an outcome has already been seen, it's probability will be non-zero. In this case, we need to increase the probability already stored there. We always increase the probability for a state by the probability of being in the prior state, times the probability of the attack outcome that caused the prior state to change to the current state. After each step, we remove the prior state from the list of outcomes.\n",
    "\n",
    "In this way, the original probabilty of being in the starting point (with value 1) is distributed over all possible outcomes. When the queue is empty, we've examined all possible outcomes that can be rearched from the starting point. \n",
    "At the end of this process, the only states that will be left with non-zero probabilities are the terminal states.\n",
    "\n",
    "Since we only ever multiplied the probabilities of independent events or added the probabilities of mutually exclusive events, the sum of the probabilities of the terminal states had better be 1 also. The code below checks that this is true. Otherwise, there would be a bug in the code."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [],
   "source": [
    "def battle_outcomes(start):\n",
    "    \"\"\"Distribution of remaining (attacker, defender) armies after a battle.\"\"\"\n",
    "    queue = deque()\n",
    "    outcomes = defaultdict(Prob)\n",
    "    queue.append(start) # starting point in the battle is only item in the queue at first\n",
    "    outcomes[start] = Prob(1) # starting point in the battle has probability 1 at first\n",
    "    while queue:\n",
    "        curr_attack = queue.popleft() # take next attack to examine out of the queue\n",
    "        prob = outcomes[curr_attack] # save probabilty of getting to this point in the battle\n",
    "        del outcomes[curr_attack] # we only want final outcomes, so remove current attack from outcomes\n",
    "        next_attack_probs = attack_outcomes(curr_attack) # look at possible outcomes for this attack\n",
    "        for next_attack in next_attack_probs:\n",
    "            new_prob = prob*next_attack_probs[next_attack] # distribute starting probabilty to outcomes\n",
    "            outcomes[next_attack] += new_prob\n",
    "            if next_attack not in queue and not finished(next_attack):\n",
    "                queue.append(next_attack) # store next attack in the queue if it might lead to other attacks\n",
    "    assert sum(outcomes.values()) == Prob(1) # this had better be true or our code is wrong\n",
    "    return ProbDist(outcomes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As I mentioned, we'll look at a lot of other exmaples of Markov chains on this site. They are frequently used in modeling sports outcomes. Hopefully this simplified example using _Risk_ will help you understand Markov chains better and motivate you to learn more about them.\n",
    "\n",
    "Now we can actually get some interesting results for _Risk_ battles.\n",
    "\n",
    "#### Checking the Results\n",
    "\n",
    "Let's first check the 2-on-1 example we did by hand before."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ProbDist({(0, 1): Prob(1237201, 13436928), (0, 2): Prob(55223, 279936), (0, 3): Prob(5957, 15552), (1, 0): Prob(883715, 13436928), (2, 0): Prob(46375, 373248), (3, 0): Prob(535, 3888)})"
      ]
     },
     "execution_count": 321,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "battle_outcomes(start)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(4402175, 13436928)"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "battle_outcomes(start).prob(attacker_wins)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Good, that matches our earlier result. Now let's look at a 3-on-1 battle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{(0, 1): 1237201/13436928, (0, 2): 55223/279936, (0, 3): 5957/15552, (1, 0): 883715/13436928, (2, 0): 46375/373248, (3, 0): 535/3888}\n"
     ]
    }
   ],
   "source": [
    "print(battle_outcomes((3, 3)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(342035, 373248)"
      ]
     },
     "execution_count": 324,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "battle_outcomes((3, 1)).prob(attacker_wins)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's see how to check this example by hand."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(95, 144)"
      ]
     },
     "execution_count": 325,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W1 = loss_probs[(3, 1)][(0, 1)]\n",
    "pA_W1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 326,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(125, 216)"
      ]
     },
     "execution_count": 326,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W2 = loss_probs[(2, 1)][(0, 1)]\n",
    "pA_W2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 327,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(5, 12)"
      ]
     },
     "execution_count": 327,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W3 = loss_probs[(1, 1)][(0, 1)]\n",
    "pA_W3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The attacker either wins on the first round, the second round or the third round. The overall probability the attacker wins a 3-on-1 battle under our assupmtions is:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Prob(342035, 373248)"
      ]
     },
     "execution_count": 328,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pA_W = pA_W1 + (1-pA_W1)*pA_W2 + (1-pA_W1)*(1-pA_W2)*pA_W3\n",
    "pA_W"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The results match, so it seems like the code is working correctly."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Probabilities of Winning a Battle\n",
    "\n",
    "Let's build a table of probabilities for all battles up to 20-on-20."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 376,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(144, 3)"
      ]
     },
     "execution_count": 376,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.DataFrame([{\n",
    "        'fichas que pueden atacar': attackers(battle),\n",
    "        'fichas totales en pais que se defiende': defenders(battle),\n",
    "        'prob_att_wins': float(battle_outcomes(battle).prob(attacker_wins))\n",
    "    } for battle in it.product(range(1, 13), range(1, 13))]\n",
    ")\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 377,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fichas que pueden atacar</th>\n",
       "      <th>fichas totales en pais que se defiende</th>\n",
       "      <th>prob_att_wins</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.417</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>0.106</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>0.018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>0.003</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fichas que pueden atacar  fichas totales en pais que se defiende  \\\n",
       "0                         1                                       1   \n",
       "1                         1                                       2   \n",
       "2                         1                                       3   \n",
       "3                         1                                       4   \n",
       "4                         1                                       5   \n",
       "\n",
       "   prob_att_wins  \n",
       "0          0.417  \n",
       "1          0.106  \n",
       "2          0.018  \n",
       "3          0.003  \n",
       "4          0.001  "
      ]
     },
     "execution_count": 377,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's pivot this table to make it more useful."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 378,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>fichas totales en pais que se defiende</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>fichas que pueden atacar</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.417</td>\n",
       "      <td>0.106</td>\n",
       "      <td>0.018</td>\n",
       "      <td>0.003</td>\n",
       "      <td>0.001</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.754</td>\n",
       "      <td>0.363</td>\n",
       "      <td>0.122</td>\n",
       "      <td>0.050</td>\n",
       "      <td>0.016</td>\n",
       "      <td>0.006</td>\n",
       "      <td>0.002</td>\n",
       "      <td>0.001</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "      <td>0.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.916</td>\n",
       "      <td>0.656</td>\n",
       "      <td>0.328</td>\n",
       "      <td>0.209</td>\n",
       "      <td>0.117</td>\n",
       "      <td>0.056</td>\n",
       "      <td>0.032</td>\n",
       "      <td>0.018</td>\n",
       "      <td>0.008</td>\n",
       "      <td>0.005</td>\n",
       "      <td>0.002</td>\n",
       "      <td>0.001</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.972</td>\n",
       "      <td>0.785</td>\n",
       "      <td>0.437</td>\n",
       "      <td>0.308</td>\n",
       "      <td>0.192</td>\n",
       "      <td>0.109</td>\n",
       "      <td>0.069</td>\n",
       "      <td>0.039</td>\n",
       "      <td>0.022</td>\n",
       "      <td>0.013</td>\n",
       "      <td>0.007</td>\n",
       "      <td>0.004</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.990</td>\n",
       "      <td>0.890</td>\n",
       "      <td>0.567</td>\n",
       "      <td>0.411</td>\n",
       "      <td>0.278</td>\n",
       "      <td>0.178</td>\n",
       "      <td>0.113</td>\n",
       "      <td>0.071</td>\n",
       "      <td>0.044</td>\n",
       "      <td>0.026</td>\n",
       "      <td>0.016</td>\n",
       "      <td>0.010</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.997</td>\n",
       "      <td>0.934</td>\n",
       "      <td>0.684</td>\n",
       "      <td>0.524</td>\n",
       "      <td>0.377</td>\n",
       "      <td>0.255</td>\n",
       "      <td>0.173</td>\n",
       "      <td>0.115</td>\n",
       "      <td>0.073</td>\n",
       "      <td>0.047</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.018</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.999</td>\n",
       "      <td>0.967</td>\n",
       "      <td>0.755</td>\n",
       "      <td>0.606</td>\n",
       "      <td>0.462</td>\n",
       "      <td>0.332</td>\n",
       "      <td>0.238</td>\n",
       "      <td>0.163</td>\n",
       "      <td>0.110</td>\n",
       "      <td>0.074</td>\n",
       "      <td>0.048</td>\n",
       "      <td>0.031</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.980</td>\n",
       "      <td>0.816</td>\n",
       "      <td>0.683</td>\n",
       "      <td>0.542</td>\n",
       "      <td>0.410</td>\n",
       "      <td>0.304</td>\n",
       "      <td>0.219</td>\n",
       "      <td>0.155</td>\n",
       "      <td>0.106</td>\n",
       "      <td>0.072</td>\n",
       "      <td>0.049</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.990</td>\n",
       "      <td>0.870</td>\n",
       "      <td>0.748</td>\n",
       "      <td>0.616</td>\n",
       "      <td>0.486</td>\n",
       "      <td>0.373</td>\n",
       "      <td>0.280</td>\n",
       "      <td>0.203</td>\n",
       "      <td>0.145</td>\n",
       "      <td>0.103</td>\n",
       "      <td>0.070</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.994</td>\n",
       "      <td>0.901</td>\n",
       "      <td>0.798</td>\n",
       "      <td>0.681</td>\n",
       "      <td>0.555</td>\n",
       "      <td>0.442</td>\n",
       "      <td>0.341</td>\n",
       "      <td>0.257</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.137</td>\n",
       "      <td>0.097</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.997</td>\n",
       "      <td>0.927</td>\n",
       "      <td>0.843</td>\n",
       "      <td>0.736</td>\n",
       "      <td>0.619</td>\n",
       "      <td>0.507</td>\n",
       "      <td>0.403</td>\n",
       "      <td>0.313</td>\n",
       "      <td>0.238</td>\n",
       "      <td>0.177</td>\n",
       "      <td>0.130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>1.000</td>\n",
       "      <td>0.998</td>\n",
       "      <td>0.949</td>\n",
       "      <td>0.877</td>\n",
       "      <td>0.784</td>\n",
       "      <td>0.678</td>\n",
       "      <td>0.569</td>\n",
       "      <td>0.465</td>\n",
       "      <td>0.370</td>\n",
       "      <td>0.288</td>\n",
       "      <td>0.221</td>\n",
       "      <td>0.165</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "fichas totales en pais que se defiende    1     2     3     4     5     6   \\\n",
       "fichas que pueden atacar                                                     \n",
       "1                                      0.417 0.106 0.018 0.003 0.001 0.000   \n",
       "2                                      0.754 0.363 0.122 0.050 0.016 0.006   \n",
       "3                                      0.916 0.656 0.328 0.209 0.117 0.056   \n",
       "4                                      0.972 0.785 0.437 0.308 0.192 0.109   \n",
       "5                                      0.990 0.890 0.567 0.411 0.278 0.178   \n",
       "6                                      0.997 0.934 0.684 0.524 0.377 0.255   \n",
       "7                                      0.999 0.967 0.755 0.606 0.462 0.332   \n",
       "8                                      1.000 0.980 0.816 0.683 0.542 0.410   \n",
       "9                                      1.000 0.990 0.870 0.748 0.616 0.486   \n",
       "10                                     1.000 0.994 0.901 0.798 0.681 0.555   \n",
       "11                                     1.000 0.997 0.927 0.843 0.736 0.619   \n",
       "12                                     1.000 0.998 0.949 0.877 0.784 0.678   \n",
       "\n",
       "fichas totales en pais que se defiende    7     8     9     10    11    12  \n",
       "fichas que pueden atacar                                                    \n",
       "1                                      0.000 0.000 0.000 0.000 0.000 0.000  \n",
       "2                                      0.002 0.001 0.000 0.000 0.000 0.000  \n",
       "3                                      0.032 0.018 0.008 0.005 0.002 0.001  \n",
       "4                                      0.069 0.039 0.022 0.013 0.007 0.004  \n",
       "5                                      0.113 0.071 0.044 0.026 0.016 0.010  \n",
       "6                                      0.173 0.115 0.073 0.047 0.030 0.018  \n",
       "7                                      0.238 0.163 0.110 0.074 0.048 0.031  \n",
       "8                                      0.304 0.219 0.155 0.106 0.072 0.049  \n",
       "9                                      0.373 0.280 0.203 0.145 0.103 0.070  \n",
       "10                                     0.442 0.341 0.257 0.190 0.137 0.097  \n",
       "11                                     0.507 0.403 0.313 0.238 0.177 0.130  \n",
       "12                                     0.569 0.465 0.370 0.288 0.221 0.165  "
      ]
     },
     "execution_count": 378,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob_table = df.pivot(index='fichas que pueden atacar', columns='fichas totales en pais que se defiende', values='prob_att_wins')\n",
    "prob_table"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Some Charts\n",
    "\n",
    "Rather than use a table, let's create some visualiztions of the results to get some more intuition.\n",
    "\n",
    "#### Heatmap\n",
    "\n",
    "Here's a [heatmap](https://en.wikipedia.org/wiki/Heat_map) of the battle probabilities. Stronger red means higher probability that the attacker wins, while stronger blue means the defender has the advantage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 381,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x864 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(16, 12))\n",
    "sns.heatmap(prob_table, square=True, annot=True, center=0.5, cmap='coolwarm', fmt='.0%', linewidths=0.1, linecolor='white',cbar=False, ax=ax)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can see that for small battles, you need to have more attackers than defenders to have better than 50-50 odds in _Risk_. This is because the defender wins on ties.\n",
    "\n",
    "After the attacker has 5 armies, however, the attacker has better than even odds with an equal number of armies. This is because the attacker can roll 3 dice, while the defender can roll only 2.\n",
    "\n",
    "#### Battle Win Probabilities by Matchup\n",
    "\n",
    "Another way to look at the results is to plot the win probabilities of various matchups. Here's a plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 333,
   "metadata": {},
   "outputs": [],
   "source": [
    "prob_curves = df[df['def'].isin([1, 2, 3, 4, 5, 10, 15, 20])].pivot(\n",
    "    index='att',\n",
    "    columns='def',\n",
    "    values='prob_att_wins'\n",
    ")\n",
    "prob_curves.columns.name = 'defenders'\n",
    "prob_curves.index.name = 'attackers'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(9, 6))\n",
    "ax = prob_curves.plot.line(\n",
    "    ax=ax,\n",
    "    xticks=prob_curves.index,\n",
    ")\n",
    "ax.set_ylabel('battle win probability')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This plot is interesting. You can see how the slope of the probability curves change as the number of attackers increases. When the number of attackers equals the number of defenders, the curves look like relatively straight lines. When the number of attackers is much less than the number of defenders, the curves start off more horizontal, then start sloping up steeply. Of course, when the number of attackers is much larger than the number of defenders, the attacker is almost certain to win the battle. This means that the probability curve has to be pretty close to horizontal near the value 1.\n",
    "\n",
    "#### Battle Win Probabilities by Attacker Advantage\n",
    "\n",
    "Here's another interesting way to visualize the _Risk_ battle probabilities. Let's call how many more armies the attacker has than the defender the _attacker advantage_. We can look at how much the attacker advantage impacts the battle win probability, as we vary the number of attacker armies.\n",
    "\n",
    "First, let's define a small function to extract the win probabilities for a given attacker advantage, and return them as a column. Notice that in our original table of win probabilities, the values with constant attacker advantage are on the _diagonal_ going down and to the right. The values with attacker advantage of 1 are, for example, (2, 1), (3, 2), (4, 3) and so on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 335,
   "metadata": {},
   "outputs": [],
   "source": [
    "def att_prob_by_advantage(df, n):\n",
    "    return (\n",
    "        df.loc[df['att'] - df['def'] == n, ['att', 'prob_att_wins']]\n",
    "        .rename(columns={'prob_att_wins': n})\n",
    "        .set_index('att')\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that we have this function, we can use it to create a new table lined up the way we need. Notice that this table has `nan` (meaning \"not-a-number' in `numpy` and `pandas`) for matchups that don't make sense. There's no way to have a 1-on-1 battle with an attacker advantage of 1, for example. Fortunately, `pandas` will ignore these `nan` values in making the plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 336,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>attacker advantage</th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>attackers</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.417</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.363</td>\n",
       "      <td>0.754</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.328</td>\n",
       "      <td>0.656</td>\n",
       "      <td>0.916</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.308</td>\n",
       "      <td>0.437</td>\n",
       "      <td>0.785</td>\n",
       "      <td>0.972</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.278</td>\n",
       "      <td>0.411</td>\n",
       "      <td>0.567</td>\n",
       "      <td>0.890</td>\n",
       "      <td>0.990</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.255</td>\n",
       "      <td>0.377</td>\n",
       "      <td>0.524</td>\n",
       "      <td>0.684</td>\n",
       "      <td>0.934</td>\n",
       "      <td>0.997</td>\n",
       "      <td>nan</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.238</td>\n",
       "      <td>0.332</td>\n",
       "      <td>0.462</td>\n",
       "      <td>0.606</td>\n",
       "      <td>0.755</td>\n",
       "      <td>0.967</td>\n",
       "      <td>0.999</td>\n",
       "      <td>nan</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.219</td>\n",
       "      <td>0.304</td>\n",
       "      <td>0.410</td>\n",
       "      <td>0.542</td>\n",
       "      <td>0.683</td>\n",
       "      <td>0.816</td>\n",
       "      <td>0.980</td>\n",
       "      <td>1.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.203</td>\n",
       "      <td>0.280</td>\n",
       "      <td>0.373</td>\n",
       "      <td>0.486</td>\n",
       "      <td>0.616</td>\n",
       "      <td>0.748</td>\n",
       "      <td>0.870</td>\n",
       "      <td>0.990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>0.190</td>\n",
       "      <td>0.257</td>\n",
       "      <td>0.341</td>\n",
       "      <td>0.442</td>\n",
       "      <td>0.555</td>\n",
       "      <td>0.681</td>\n",
       "      <td>0.798</td>\n",
       "      <td>0.901</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>0.177</td>\n",
       "      <td>0.238</td>\n",
       "      <td>0.313</td>\n",
       "      <td>0.403</td>\n",
       "      <td>0.507</td>\n",
       "      <td>0.619</td>\n",
       "      <td>0.736</td>\n",
       "      <td>0.843</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>0.165</td>\n",
       "      <td>0.221</td>\n",
       "      <td>0.288</td>\n",
       "      <td>0.370</td>\n",
       "      <td>0.465</td>\n",
       "      <td>0.569</td>\n",
       "      <td>0.678</td>\n",
       "      <td>0.784</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>0.155</td>\n",
       "      <td>0.205</td>\n",
       "      <td>0.267</td>\n",
       "      <td>0.342</td>\n",
       "      <td>0.427</td>\n",
       "      <td>0.524</td>\n",
       "      <td>0.626</td>\n",
       "      <td>0.728</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>0.145</td>\n",
       "      <td>0.191</td>\n",
       "      <td>0.248</td>\n",
       "      <td>0.315</td>\n",
       "      <td>0.395</td>\n",
       "      <td>0.484</td>\n",
       "      <td>0.579</td>\n",
       "      <td>0.678</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>0.136</td>\n",
       "      <td>0.179</td>\n",
       "      <td>0.230</td>\n",
       "      <td>0.293</td>\n",
       "      <td>0.366</td>\n",
       "      <td>0.448</td>\n",
       "      <td>0.538</td>\n",
       "      <td>0.631</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>0.128</td>\n",
       "      <td>0.167</td>\n",
       "      <td>0.215</td>\n",
       "      <td>0.272</td>\n",
       "      <td>0.339</td>\n",
       "      <td>0.416</td>\n",
       "      <td>0.500</td>\n",
       "      <td>0.589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>0.120</td>\n",
       "      <td>0.156</td>\n",
       "      <td>0.201</td>\n",
       "      <td>0.253</td>\n",
       "      <td>0.316</td>\n",
       "      <td>0.387</td>\n",
       "      <td>0.465</td>\n",
       "      <td>0.550</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>0.113</td>\n",
       "      <td>0.147</td>\n",
       "      <td>0.187</td>\n",
       "      <td>0.236</td>\n",
       "      <td>0.294</td>\n",
       "      <td>0.360</td>\n",
       "      <td>0.434</td>\n",
       "      <td>0.514</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>0.107</td>\n",
       "      <td>0.137</td>\n",
       "      <td>0.175</td>\n",
       "      <td>0.221</td>\n",
       "      <td>0.274</td>\n",
       "      <td>0.336</td>\n",
       "      <td>0.405</td>\n",
       "      <td>0.480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>0.100</td>\n",
       "      <td>0.129</td>\n",
       "      <td>0.165</td>\n",
       "      <td>0.206</td>\n",
       "      <td>0.256</td>\n",
       "      <td>0.314</td>\n",
       "      <td>0.379</td>\n",
       "      <td>0.450</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "attacker advantage     0     1     2     3     4     5     6     7\n",
       "attackers                                                         \n",
       "1                  0.417   nan   nan   nan   nan   nan   nan   nan\n",
       "2                  0.363 0.754   nan   nan   nan   nan   nan   nan\n",
       "3                  0.328 0.656 0.916   nan   nan   nan   nan   nan\n",
       "4                  0.308 0.437 0.785 0.972   nan   nan   nan   nan\n",
       "5                  0.278 0.411 0.567 0.890 0.990   nan   nan   nan\n",
       "6                  0.255 0.377 0.524 0.684 0.934 0.997   nan   nan\n",
       "7                  0.238 0.332 0.462 0.606 0.755 0.967 0.999   nan\n",
       "8                  0.219 0.304 0.410 0.542 0.683 0.816 0.980 1.000\n",
       "9                  0.203 0.280 0.373 0.486 0.616 0.748 0.870 0.990\n",
       "10                 0.190 0.257 0.341 0.442 0.555 0.681 0.798 0.901\n",
       "11                 0.177 0.238 0.313 0.403 0.507 0.619 0.736 0.843\n",
       "12                 0.165 0.221 0.288 0.370 0.465 0.569 0.678 0.784\n",
       "13                 0.155 0.205 0.267 0.342 0.427 0.524 0.626 0.728\n",
       "14                 0.145 0.191 0.248 0.315 0.395 0.484 0.579 0.678\n",
       "15                 0.136 0.179 0.230 0.293 0.366 0.448 0.538 0.631\n",
       "16                 0.128 0.167 0.215 0.272 0.339 0.416 0.500 0.589\n",
       "17                 0.120 0.156 0.201 0.253 0.316 0.387 0.465 0.550\n",
       "18                 0.113 0.147 0.187 0.236 0.294 0.360 0.434 0.514\n",
       "19                 0.107 0.137 0.175 0.221 0.274 0.336 0.405 0.480\n",
       "20                 0.100 0.129 0.165 0.206 0.256 0.314 0.379 0.450"
      ]
     },
     "execution_count": 336,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob_by_adv = pd.concat([att_prob_by_advantage(df, n) for n in range(0, 8)], axis='columns')\n",
    "prob_by_adv.columns.name = 'attacker advantage'\n",
    "prob_by_adv.index.name = 'attackers'\n",
    "prob_by_adv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 337,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 648x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plt.subplots(figsize=(9, 6))\n",
    "ax = prob_by_adv.plot(ax=ax, xticks=prob_by_adv.index)\n",
    "ax.set_ylabel('battle win probability')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The bottommost (red) curve is for an attacker advantage of zero: the even-army matchup we already looked at. You can see how it exceeds 50% when the attacker has more than 5 armies. The line continues to rise slowly as the number of attacker armies increases.\n",
    "\n",
    "With a 1- or 2-army attacker advantage, the curves drop quickly and then become relativley flat around 65-75% win probability. For _Risk_ battles with more than 5 attacking armies and a 1- or 2-army defender advantage, you can just remember the rule of thumb that the attacker should win around 70% of the battles.\n",
    "\n",
    "It seems like the curves are gradually converging as the number of armies increase. Let's see what the values are for 100 attacking armies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 338,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0016862523740338508"
      ]
     },
     "execution_count": 338,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "float(battle_outcomes((100, 100)).prob(attacker_wins))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 339,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.004692279636173258"
      ]
     },
     "execution_count": 339,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "float(battle_outcomes((107, 100)).prob(attacker_wins))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "So yes, the curves are closer together when the number of armies gets very large. The even-matchup curve is up over 80% win probabilty, while the curve for 7 attacker advantage is down to 91.5%, not much less than it is at 20 armies in the plot above.\n",
    "\n",
    "### Closing Thoughts\n",
    "\n",
    "For the larger attacker advantages in the plot above, the curves slope down over the entire range we examined here. This means that on a relative basis, your win probability is getting _worse_ as your throw more attackers into the battle. The analysis says you have better odds of success attacking 15-on-8 compared with 20-on-13.\n",
    "\n",
    "There's clearly more going on here than meets the eye. It turns out that win probability isn't the only way (or the best way) to analyze _Risk_, since it ignores how many armies the attacker can expect to have left over even if she wins. If you have a high probability of winning an attack, but will lose an enormous number of armies in the process, the attack might not make sense.\n",
    "\n",
    "We'll look at these issues in more detail in future posts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
