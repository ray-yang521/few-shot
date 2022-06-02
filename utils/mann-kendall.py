import pymannkendall as mk
import pandas as pd
import os


def annual_avg(d):
    d["annual_count"] = d.index // 12
    ds = d.groupby("annual_count").mean()
    return ds


def season_avg(d):
    d["season_count"] = d.index // 3
    d_spring = d[(d["season_count"] % 4 == 0)]
    d_summer = d[(d["season_count"] % 4 == 1)]
    d_autumn = d[(d["season_count"] % 4 == 2)]
    d_winter = d[(d["season_count"] % 4 == 3)]

    ds_spring = d_spring.groupby("season_count").mean()
    ds_summer = d_summer.groupby("season_count").mean()
    ds_autumn = d_autumn.groupby("season_count").mean()
    ds_winter = d_winter.groupby("season_count").mean()
    return ds_spring, ds_summer, ds_autumn, ds_winter


def mk_test(df, alpha):
    # the following dictionary's structure must be modified by different dataset
    print(df.columns)
    data = {}
    for each_col in df.columns:
        data[each_col] = df[each_col].values

    slope_and_z = []

    for key in data.keys():
        specific_col = data[key]
        result = mk.original_test(specific_col, alpha)
        z = result[3]
        slope = result[7]
        print(result)
        slope_and_z.append([slope, z])
    out = pd.DataFrame(slope_and_z, index=data.keys(), columns=["slope", "z"])
    return out


if __name__ == '__main__':
    # create directory for saving if not exist
    target_path = ""
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Data generation for analysis
    excel_path = ""

    df = pd.read_excel(excel_path)[50:400]

    alpha = 0.05

    annual_avg_data = annual_avg(df)
    mk_test(annual_avg_data, alpha).to_excel(os.path.join(target_path, "annual_avg_alpha=" + str(alpha) + ".xlsx"))

    season_avg_data = season_avg(df)
    spring = season_avg_data[0]
    summer = season_avg_data[1]
    autumn = season_avg_data[2]
    winter = season_avg_data[3]

    mk_test(spring, alpha).to_excel(os.path.join(target_path, "spring_alpha=" + str(alpha) + ".xlsx"))
    mk_test(summer, alpha).to_excel(os.path.join(target_path, "summer_alpha=" + str(alpha) + ".xlsx"))
    mk_test(autumn, alpha).to_excel(os.path.join(target_path, "autumn_alpha=" + str(alpha) + ".xlsx"))
    mk_test(winter, alpha).to_excel(os.path.join(target_path, "winter_alpha=" + str(alpha) + ".xlsx"))
    print("finished!")
