{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "the final code to get the results for mmdet from the tubetk-detection results\"TubeTK/linkres/res/\"\n",
    "\"\"\" \n",
    "import csv\n",
    "from collections import defaultdict\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "def load_without_columns(filePath):\n",
    "    mydict = defaultdict(list)\n",
    "    results=[]\n",
    "    with open(filePath, mode='r') as csv_file:\n",
    "        csv_reader = csv.DictReader(csv_file, fieldnames=[\"id\",\"c1\",\"c2\",\"c3\",\"c4\",\"c5\",\"c6\",\"c7\",\"c8\",\"c9\"])\n",
    "        for row in csv_reader:\n",
    "            mylist = [float(row[\"c2\"]), float(row[\"c3\"]), float(row[\"c4\"]), float(row[\"c5\"]), float(row[\"c7\"])]\n",
    "            np.array(mydict[int(row[\"id\"])].append(mylist))\n",
    "\n",
    "        for key in mydict:\n",
    "            results.append(np.array(mydict[key])) \n",
    "    return results"
   ]
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
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
