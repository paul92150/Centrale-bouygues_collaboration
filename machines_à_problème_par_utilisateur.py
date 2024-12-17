import matplotlib.pyplot as plt

# Dictionnaire des nombres de machines à problèmes par utilisateur
problematic_devices = {'Beta': 11, 'Gamma': 15, 'Lambda': 1, 'Delta': 2, 'Alpha': 4, 'Nu': 1, 'Zeta': 1, 'Epsilon': 1}

# Dictionnaire du nombre total de dispositifs par utilisateur
total_devices = {'Alpha': 50, 'Beta': 76, 'Delta': 51, 'Epsilon': 14, 'Eta': 10, 'Evil': 5, 'Gamma': 132, 'Iota': 14, 'Kappa': 14, 'Lambda': 11, 'Mu': 9, 'Nu': 15, 'Theta': 8, 'Zeta': 18}

# Calcul des proportions de dispositifs problématiques
proportions = {user: (problematic_devices.get(user, 0) / total) for user, total in total_devices.items()}

# Tri des utilisateurs par nombre de dispositifs problématiques
sorted_users_by_problematic = sorted(problematic_devices, key=problematic_devices.get, reverse=True)
sorted_problematic_values = [problematic_devices[user] for user in sorted_users_by_problematic]

# Tri des utilisateurs par proportion de dispositifs problématiques
sorted_users_by_proportion = sorted(proportions, key=proportions.get, reverse=True)
sorted_proportion_values = [proportions[user] for user in sorted_users_by_proportion]

# Listes pour les valeurs triées
sorted_total_values = [total_devices[user] for user in sorted_users_by_proportion]

# Création de la figure et des sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Graphique à barres pour le nombre de dispositifs problématiques (trié)
ax1.bar(sorted_users_by_problematic, sorted_problematic_values, color='skyblue')
ax1.set_title('number of problematic devices per user')
ax1.set_xlabel('User')
ax1.set_ylabel('Number of problematic devices')
for i, value in enumerate(sorted_problematic_values):
    ax1.text(i, value + 0.2, str(value), ha='center')

# Graphique à barres pour les proportions de dispositifs problématiques (trié)
ax2.bar(sorted_users_by_proportion, sorted_proportion_values, color='lightgreen')
ax2.set_title('Ratio of problematic devices per user')
ax2.set_xlabel('User')
ax2.set_ylabel('Proportion of problematic devices')
ax2.set_xticklabels(sorted_users_by_proportion, rotation=45)
for i, value in enumerate(sorted_proportion_values):
    ax2.text(i, value + 0.01, f'{value:.2%}', ha='center')

# Ajustement de l'affichage
plt.tight_layout()
plt.show()
