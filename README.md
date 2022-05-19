# text_based-artwork_recommender
Someone asked for my help with building a text-based artwork recommendation system. Two potentially useful data fields are the artwork name and the artwork description, the latter often missing (the attached data sample only includes these two columns. 

This is my simple-stupid solution for recommending artworks based on the most similar titles/descriptions, using Glove word embeddings.

Example output:

- for the artiwork entitled "Starry sky with yellow and clouds", top suggestions would be 'fence and broken trees with pink sky', 'colors of grey', 'three tall trees and hurricane sky','colors of grey #2', 'a woman in yellow hat', 'moon night of yehliu 58x120cm' - looks pretty adequate to me!

- for the artwork with description "Juan Usl paintings are complex interactions and incorporate a great diversity of art historical references sensory and mental impressions various pictorial languages the gesture of painting and how the matter paints itself functions The first step is the canvas s preparation with multiple layers of gesso which is a key element that will remain visible This continuous manifestation of the gesso also conveys a philosophical resonance for Usl namely that the beginning is present at the end that the painting is a self contained entity a complete object not merely within its four sides but in the vertical layering of its surface as well The process of painting consists of a natural harmony between the manual act and the intellectual decisions Movement or even better displacement is a thematic key in his work In Usl s work we can read a constant dichotomy between opposing and complementary elements at the same time order and chaos presence and absence flatness and depth Most of his paintings are a juxtaposition of color areas and lines structures that seem to come and go like the fragments of a story The Artist presents his work as a temporary delimitation of infinite surfaces or as fragments from an infinite structure of lines Particularly well known is the series of paintings called So que Revelabas Dream that revelead In these works the artist through a deeply introspective practice seems to give shape to his more intimate self While painting Usl tries to connect rhythmically with his palpitation making each stroke a symbolic representation of the beating of his heart return the theme of disorientation no longer understood only in a physical sense but also and above all in a more temporal and perceptive way His paintings take the viewer into a labyrinthine space in which the articulation seems to indicate a specific direction while paradoxically it leaves open the way to interpretation", top 3 suggestinos are:
       - 'thomas sch tte s b 1954 germany work develops in many different areas architectural models installations watercolors banners sculptures in different scales etchings are some of the most expressive mediums the artist uses during the last years sch tte generates a personal collection of techniques and styles from which draws its material every time he believes that form material and color have their own language which is impossible to translate a few of the issues that concern the artist and are subjects of his work are the problems that contemporary man faces with oneself living with others confronting the present facing his mortality his work is often ironic and critical at the same time '
       - 'mark manders born in 1968 in volkel nl lives and works in ronse be the work of mark manders resembles a fictional building divided into separate rooms and levels of which the size and shape can never exactly be determined potential shifts and extensions constantly threaten the cohesion of the ever expanding self portrait manders works toward one big overarching moment that will bring together all his works continuously interconnected and in dialogue with each other as a sculptor manders adheres to the tradition of bronze sculpture yet also incorporates contemporary materials in his work blurring the line between reality and illusion it often becomes difficult to distinguish when manders is actually integrating natural wood or just a painted wood imitation this also applies to the androgynous figures or faces that seem to have been fashioned out of wet clay creating the impression that they just left the artist s studio or conversely were abandoned by the artist mid work the illusion of peeling dry clay creates a sense of foreboding as if the sculpture could crumble into fine dust and disappear at any time there appears to be a definite separation between the sculpture and the person who realized it as if it was abandoned by its creator or could not be completed '
       - 'ida tursic wilfried mille motivated by rather unconventional attitude to their peers are sharing a strong passion for the pictorial tradition starting from a database patiently built over these years of 140 000 images grouped by topic the artists mix a more classical tradition with contemporaneity the result is a strong and always various mix of techniques and subject hard to label abstract and figurative melt together into the same subject leaving the spectator free to focus on every single detail as if they were a separate part in a larger and more complex story '
