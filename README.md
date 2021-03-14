# Mechanical Regression for Supervised Learning
This is the Bachelor's thesis I wrote for my B.Sc. degree in
Mathematics during the winter term 2020/21.
In the thesis, I examined mechanical regression which was proposed by Houman Owhadi 
in the paper "Do Ideas Have Shape? Plato's Theory of Forms as the Continuous Limit of
Artificial Neural Networks" [1].
Mechanical regression is a method for supervised learning that was derived from a 
theoretical model of residual neural networks (ResNets).

This repository contains an implementation of the algorithm suggested in
section 3 in [1] (in the `python` folder) and also the LaTeX code for the thesis and 
the figures.

The following are the abstracts in English and German.

## Abstract
Although neural networks have been used with extraordinary success for many years in machine learning, they are still not fully understood from a mathematical point of view.
In "Do Ideas Have Shape? Plato's Theory of Forms as the Continuous Limit of Neural Networks", Houman Owhadi analyzes a subclass of neural networks which are called residual neural networks (ResNets) and shows their convergence towards a continuous mechanical system.
This system resembles algorithms from image registration and computational anatomy.

In this thesis, parts of Owhadi's results are presented and discussed.
It is shown that ResNets relate to a discretized stationary action principle which can be formulated as a discrete geodesic shooting problem.
Because of this connection to physics, these equivalent problems are summarized as _mechanical regression_.
All three discrete problems have continuous counterparts and converge towards them.
This convergence is in the sense that as the number of ResNet layers tends towards infinity or the step size towards zero, the minimal values converge and the adherence points of sequences of minimizers are solutions to the continuous problems.

From the continuous geodesic shooting problem, an algorithm for the supervised learning problem can be derived.
Parts of this algorithm are implemented and numerical experiments are conducted.
The results closely resemble those presented by Owhadi.	

## Zusammenfassung
Obwohl neuronale Netze seit mehreren Jahren mit außergewöhnlichem Erfolg im maschinellen Lernen eingesetzt werden, sind sie aus mathematischer Perspektive immer noch nicht vollständig verstanden.
In "Do Ideas Have Shape? Plato's Theory of Forms as the Continuous Limit of Neural Networks" analysiert Houman Owhadi eine Unterklasse neuronaler Netze, welche "Residual Neural Networks" (ResNets) genannt werden und zeigt, dass diese gegen ein stetiges mechanisches System konvergieren.
Dieses System ähnelt Algorithmen aus dem Bereich der Bilderfassung und rechenbasierter Anatomie.

In dieser Arbeit werden Teile von Owhadis Ergebnissen präsentiert und diskutiert.
Es wird gezeigt, dass ResNets in Zusammenhang zu einem diskreten Prinzip der stationären Wirkung stehen, welche als diskretes Geodesic-Shooting-Problem formuliert werden kann.
Aufgrund dieser Verbindung zur Physik werden diese äquivalenten Probleme als _Mechanische Regression_ zusammengefasst.
Alle drei diskreten Probleme haben stetige Gegenstücke und konvergieren gegen diese.
Diese Konvergenz ist folgendermaßen zu verstehen:
Strebt die Anzahl der ResNet-Schichten gegen unendlich beziehungsweise die Schrittweite gegen Null, konvergieren die minimalen Funktionswerte und die Berührpunkte von Folgen von Minimieren sind Lösungen der stetigen Probleme.

Aus der Formulierung als stetiges Geodesic-Shooting-Problem kann ein Algorithmus für das überwachte Lernen abgeleitet werden.
Teile dieses Algorithmus werden implementiert und numerische Experimente werden durchgeführt.
Die Ergebnisse sind denen, die von Owhadi berichtet wurden, sehr ähnlich.

## References

[1] Houman Owhadi. Do Ideas Have Shape? Plato's Theory of Forms as the Continuous Limit of
Artificial Neural Networks. ArXiv, https://arxiv.org/abs/2008.03920.
