"""Kleine Testkoerper (alle geschlossen, nach aussen orientiert)."""
import numpy as np

import kkmeshcut as mc


def cube(a=1.0, lo=(0, 0, 0), sub=1):
    """Wuerfel; sub>1 unterteilt jede Seite in sub*sub Quadrate (gerade Eckpunkte entlang der Kanten)."""
    lo = np.asarray(lo, float)
    tris = []
    n = sub
    for axis in range(3):
        for side in (0, 1):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            for i in range(n):
                for j in range(n):
                    def pt(ii, jj):
                        p = np.zeros(3)
                        p[axis] = side * a
                        p[u] = ii * a / n
                        p[v] = jj * a / n
                        return p + lo
                    p00, p10, p11, p01 = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
                    quad = [(p00, p10, p11), (p00, p11, p01)]
                    for t in quad:
                        # Umlaufsinn so, dass die Normale nach aussen zeigt
                        nrm = np.cross(t[1] - t[0], t[2] - t[0])
                        out = np.zeros(3)
                        out[axis] = 1 if side else -1
                        tris.append(t if nrm @ out > 0 else (t[0], t[2], t[1]))
    V, F = mc.weld_triangles(np.array(tris))
    return mc.Mesh(V, F, "cube")


def sphere(n=12, m=8, r=1.0, c=(0, 0, 0)):
    c = np.asarray(c, float)
    V = [np.array([0, 0, r])]
    for j in range(1, m):
        th = np.pi * j / m
        for i in range(n):
            ph = 2 * np.pi * i / n
            V.append(r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    V.append(np.array([0, 0, -r]))
    V = np.array(V) + c
    F = []
    for i in range(n):
        F.append((0, 1 + i, 1 + (i + 1) % n))
    for j in range(m - 2):
        for i in range(n):
            a, b = 1 + j * n + i, 1 + j * n + (i + 1) % n
            c2, d = a + n, b + n
            F.append((a, c2, b)); F.append((b, c2, d))
    last = len(V) - 1
    for i in range(n):
        a, b = 1 + (m - 2) * n + i, 1 + (m - 2) * n + (i + 1) % n
        F.append((a, last, b))
    return mc.Mesh(V, np.array(F), "sphere")


def torus(R=2.0, r=0.7, n=24, m=12, c=(0, 0, 0)):
    c = np.asarray(c, float)
    V = []
    for i in range(n):
        th = 2 * np.pi * i / n
        for j in range(m):
            ph = 2 * np.pi * j / m
            V.append([(R + r * np.cos(ph)) * np.cos(th), (R + r * np.cos(ph)) * np.sin(th), r * np.sin(ph)])
    V = np.array(V) + c
    F = []
    for i in range(n):
        for j in range(m):
            a = i * m + j
            b = ((i + 1) % n) * m + j
            c2 = ((i + 1) % n) * m + (j + 1) % m
            d = i * m + (j + 1) % m
            F.append((a, b, c2)); F.append((a, c2, d))
    mesh = mc.Mesh(V, np.array(F), "torus")
    if mc.volume_area(mesh.V, mesh.F)[0] < 0:
        mesh = mc.Mesh(mesh.V, mesh.F[:, [0, 2, 1]], "torus")
    return mesh


def prism(poly, h=1.0):
    """Gerade Saeule ueber einem Polygon (gegen den Uhrzeigersinn, ohne Loecher, beliebig nichtkonvex, per Ohrenschnitt)."""
    poly = [tuple(p) for p in poly]
    n = len(poly)
    P2 = {i: poly[i] for i in range(n)}
    tris, _ = mc.ear_clip(P2, list(range(n)))
    V = np.array([(x, y, 0.0) for x, y in poly] + [(x, y, h) for x, y in poly])
    F = [(a, c, b) for a, b, c in tris] + [(a + n, b + n, c + n) for a, b, c in tris]
    for i in range(n):
        j = (i + 1) % n
        F.append((i, j, j + n)); F.append((i, j + n, i + n))
    return mc.Mesh(V, np.array(F), "prism")


L_SHAPE = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
U_SHAPE = [(0, 0), (5, 0), (5, 4), (4, 4), (4, 1), (1, 1), (1, 4), (0, 4)]
