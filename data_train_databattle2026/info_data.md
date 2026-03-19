# Position des aéroports

Ces données ont été observées par Météorage sur une période de 10 ans
(\'2016-01-01 00:00:00\' and \'2025-12-31 23:59:59\') et dans un rayon
de 30 km autour de l'aéroport dont la position en degrés décimaux WGS84
(EPSG :4326) est indiquée ci-dessous.

> Bron : longitude = 4.9389, latitude = 45.7294
>
> Bastia : longitude = 9.4837, latitude = 42.5527
>
> Ajaccio : longitude = 8.8029, latitude = 41.9236
>
> Nantes : longitude = -1.6107, latitude = 47.1532
>
> Pise : longitude = 10.399, latitude = 43.695
>
> Biarritz : longitude = -1.524, latitude = 43.4683

# Description des champs

**Chaque ligne correspond à un éclair dont les caractéristiques sont
décrites dans le paragraphe suivant.**

**date** : date en UTC (Universal Time Coordinated) à laquelle l'éclair
s'est produit

**lon/lat** : position de l'éclair en longitude et latitude en degrés
décimaux (EPSG :4326)

**amplitude** : polarité et intensité maximale du courant de décharge
ayant circulé dans l'éclair.

**maxis** : erreur de localisation théorique estimée en km.

**icloud** : booléen indiquant la nature de l'éclair.

False = éclair nuage-sol (contact au sol)

True = éclair intra-nuage (pas de contact au sol)

**dist** : distance de l'éclair par apport à l'aéroport

**azimuth** : direction en degrés de la position de l'éclair par rapport
à l'aéroport (0 degré indique le nord, 90 degrés indiquent l'est, 180
degrés indiquent le sud et 270 degrés indiquent l'ouest).

**lightning_id** : identifiant d\'un éclair dans le fichier

**lightning_airport_id** : identifiant d\'un éclair au sein d\'un
aéroport.

**alert_airport_id** : indique le numéro de l\'alerte à laquelle
appartient cet éclair au sien d\'un aéroport.

**is_last_lightning_cloud_ground** : est un booléen qui vaut True quand
il correspond au dernier éclair nuage-sol d\'une alerte

Les deux dernières colonnes ne sont remplis que pour les éclairs à moins
de 20km d'un aérport.

**NB :** l'aéroport de Pise a eu un système d'enregistrement différent
pour l'année 2016 concernant les éclairs intra-nuage (à confirmer avec
meteorage) . Aussi, peut-être il prudent d'écarter ces données pour ce
type d'analyse.
