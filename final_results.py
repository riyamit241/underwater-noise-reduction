import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Riya\Desktop\sem 6 study material\CVPR\CVPR lab\case study\raw-890\filter_results.csv")  # Replace with your filename

#  average PSNR for filters
avg_psnr = {
    'Gaussian': df['PSNR_Gaussian'].mean(),
    'Median': df['PSNR_Median'].mean(),
    'Bilateral': df['PSNR_Bilateral'].mean()
}

# best filter
best_filter = max(avg_psnr, key=avg_psnr.get)

print("Average PSNR values:")
for k, v in avg_psnr.items():
    print(f"{k}: {v:.2f} dB")

print(f"\n🏆 Best performing filter: {best_filter} (based on average PSNR)")

sns.set(style="whitegrid")      # BAR PLOT
plt.figure(figsize=(8, 5))
sns.barplot(x=list(avg_psnr.keys()), y=list(avg_psnr.values()), palette='coolwarm')

plt.title("Average PSNR Comparison of Filters")
plt.ylabel("Average PSNR (dB)")
plt.xlabel("Filtering Technique")


for i, (filt, val) in enumerate(avg_psnr.items()):
    if filt == best_filter:
        plt.text(i, val + 0.2, "🏆 Best", ha='center', va='bottom', fontsize=10, color='green')

plt.ylim(min(avg_psnr.values()) - 1, max(avg_psnr.values()) + 2)
plt.tight_layout()
plt.show()