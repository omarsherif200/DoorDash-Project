import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class customMetrics:
    def __init__(self):
        pass
    def correct_observations(self, target_col, prediction_col, minutes=5):
        total_samples = len(target_col)
        error = np.array([abs(prediction_col[i] - target_col[i]) for i in range(0, total_samples)])
        observations_within_time = (error <= (minutes * 60)).sum()
        return observations_within_time

    def plot_accuracy_per_time(self, target_col, prediction_col):
        total_samples = len(target_col)
        correct_observations_within_5_minutes = self.correct_observations(target_col, prediction_col, minutes=5)
        accuracy_within_5_minutes = correct_observations_within_5_minutes / total_samples
        correct_observations_from_5to10_minutes = \
            self.correct_observations(target_col, prediction_col, minutes=10) - correct_observations_within_5_minutes

        accuracy_from_5to10_minutes = correct_observations_from_5to10_minutes / total_samples

        correct_observations_from_10to15_minutes = \
            self.correct_observations(target_col, prediction_col, minutes=15) - self.correct_observations(target_col,
                                                                                                          prediction_col,
                                                                                                          minutes=10)

        accuracy_from_10to15_minutes = correct_observations_from_10to15_minutes / total_samples

        correct_observations_from_15to20_minutes = \
            self.correct_observations(target_col, prediction_col, minutes=20) - self.correct_observations(target_col,
                                                                                                          prediction_col,
                                                                                                          minutes=15)

        accuracy_from_15to20_minutes = correct_observations_from_15to20_minutes / total_samples

        correct_observations_above_20_minutes = total_samples - self.correct_observations(target_col, prediction_col,
                                                                                          minutes=20)

        accuracy_above_20_minutes = correct_observations_above_20_minutes / total_samples

        xlabel = ['Within 5 minutes', 'From 5 to 10 minutes', 'From 10 to 15 minutes', 'From 15 to 20 minutes',
                  'Above 20 minutes']
        ylabel = [accuracy_within_5_minutes, accuracy_from_5to10_minutes, accuracy_from_10to15_minutes,
                  accuracy_from_15to20_minutes, accuracy_above_20_minutes]

        plt.bar(xlabel, ylabel)
        plt.xlabel('Time periods', fontsize=15)
        plt.ylabel("Percentage", fontsize=15)
        plt.title('Predictions within time periods', fontsize=20)
        plt.xticks(rotation=10)
        plt.show()

    def get_overestimate_percentage(self, target_col, prediction_col):
        return (prediction_col >= target_col).sum() / len(target_col)

    def plot_overestimate_underestimate_ratio(self, target_col, prediction_col):
        overestimate_ratio = self.get_overestimate_percentage(target_col, prediction_col)
        underestimate_ratio = 1 - overestimate_ratio
        labels = ['Overestimate Ratio', 'Underestimate Ratio']
        ratios = [overestimate_ratio, underestimate_ratio]
        plt.pie(ratios, labels=labels, startangle=90, autopct='%1.2f%%', textprops={'fontsize': 15})
        plt.title("Overestimate To Underestimate observations ratio", fontsize=20)
        plt.show()