# Generate HTML table with interactive 3D models
import numpy as np
import os
import sys
from collections import defaultdict
import base64

"""
The current format is the following table (1 for each mesh + anchor sample):
            Postprocess 1  Postprocess 2  Postprocess 2
Config Details   -            -                  -
3D mesh          -            -                  -
Distortion deets -            -                  -
"""
# TODO: Split mesh-anchor predictions into SEPARATE TABLES
# - Consider: separate HTML per mesh if these tables get too big
class MeshTableViewer:
    """ Class for visualizing table of mesh results """
    def __init__(self, views, stats, titles = None, columns = None, tdwidth=100):
        """ view: [method](methodconfig, [mesh][anchor](anchorpos, camerapos, [postprocess] = (selection path, nonselect path)))
            stats: [method][mesh][anchor][postprocess][statname] = statvalue
            titles: [method] = title
            columns: defaults to postprocess
            tdwidth: % of table to assign to each column """
        self.views = views
        self.stats = stats
        self.tdwidth = tdwidth

        # Set column names dictionary
        # NOTE: If None, then defaults to postprocess keys of views dictionary
        self.columns = columns

        # Set title
        self.titles = titles

        # Set HTML string
        self.html = {}
        self.sethtml()

    def sethtml(self):
        # New html string for each method
        for methodname, (methodconfig, meshdict) in self.views.items():
            # Sometimes will get methods in views without stats
            if methodname not in self.stats.keys():
                print(f"Warning: {methodname} stats not found. Skipping ... ")
                continue

            # Init document with style
            html = "<!DOCTYPE html>\n<html>\n"
            html += '<link rel="stylesheet" type="text/css" />\n'
            html += "<style>\n"
            html += ("html, body {\n"
                        "margin:0px;\n"
                        "height:100%;\n"
                    "}")
            html += (".center {\n"
                    "   text-align: center;\n"
                    "}\n")
            html += ("canvas {\n"
                    "width: 100%;\n"
                    "height: 100%;\n"
                    "display: block;\n"
                    "}\n"
                    )
            # html += (".td {\n"
            #         f"width:{self.tdwidth}%;\n"
            #         f"height:{self.tdwidth}%;\n"
            #         "}\n"
            #         )
            html += ("*[data-selectionobj] {\n"
                    "display: inline-block;\n"
                    "width:100%;\n"
                    "height:100%;\n"
                    "}\n"
                    )
            html += "</style>\n"
            html += (f"</head>\n"
                    f"<body>\n"
                    )

            # 3JS Scripts
            html += (
                "<script src='./scripts/three.js'></script>\n"
                "<script src='./scripts/OBJLoader.js'></script>\n"
                "<script src='./scripts/OrbitControls.js'></script>\n"
                "<script src='./scripts/render.js'></script>\n"
            )

            # Add title
            if self.titles is not None:
                html += f"<h1>{self.titles[methodname]}</h1>\n"

            # New table for each meshno
            for meshname, anchordict in meshdict.items():
                for anchorno, (anchorpos, camerapos, ppdict) in anchordict.items():
                        # Init table with headers
                        meshno = f"{meshname}_{anchorno}"
                        html += "<table style='width:100%;'>\n"
                        html += f"<caption style='font-size:24px;'>{meshno}</caption>\n"
                        html += "  <tr>\n"
                        if self.columns:
                            for column in self.columns:
                                html += f"    <th style='text-align:center;white-space:nowrap;width:{self.tdwidth}%'>{column}</th>\n"
                        else:
                            for postprocess in ppdict.keys():
                                html += f"    <th style='text-align:center;white-space:nowrap;width:{self.tdwidth}%'>{postprocess}</th>\n"
                        html += "  </tr>\n"

                        # Separate string per table row
                        configstr = "  <tr>\n"
                        objstr = "  <tr>\n"
                        distortstr = "  <tr>\n"
                        for postprocess, (selectpath, nonselectpath) in ppdict.items():
                            # First row is the configuration details
                            configstr += f'   <td style="width:{self.tdwidth}%"><span style="white-space: pre-wrap">{methodconfig}</span></td>\n'

                            # If meshno begins with number, then need to add a letter in front
                            # if meshno[0].isdigit():
                            #     querymesh = f"A{meshno}"
                            # else:
                            #     querymesh = meshno

                            # Second row is the obj paths
                            objstr += (f'<td style="width:{self.tdwidth}%"><span data-selectionobj="{selectpath}" data-nonselectionobj="{nonselectpath}"'
                                        f' data-anchorx={anchorpos[0]:0.5f} data-anchory={anchorpos[1]:0.5f} data-anchorz={anchorpos[2]:0.5f}'
                                        f' data-camerax={camerapos[0]:0.5f} data-cameray={camerapos[1]:0.5f} data-cameraz={camerapos[2]:0.5f}></span></td>\n')

                            # Third row is the distortion metrics
                            if self.stats is not None:
                                stats = self.stats[methodname][meshname][anchorno][postprocess]
                                distortstr += f"  <td style='white-space:nowrap;width:{self.tdwidth}%'>\n"
                                distortstr += "       <ul>\n"
                                for statname, value in stats.items():
                                    if isinstance(value, str):
                                        distortstr += f"<li>{statname}: {value}</li>\n"
                                    elif "distortion" in statname:
                                        distortstr += f"<li>{statname}: {value:.3e}</li>\n"
                                    else:
                                        # Default: regular float formatting
                                        distortstr += f"<li>{statname}: {value:.3f}</li>\n"
                                distortstr += "      </ul>\n</td>\n"
                        # Put everything back together
                        html += configstr + "</tr>\n" + objstr + "</tr>\n" + distortstr + "</tr>\n"
                        html += " </table>\n"

            # Key script for off-screen rendering
            html += "<script src='./scripts/globalcanvas.js'></script>\n"

            html += "</body>\n</html>\n"

            # Set html dictionary with key as the file savename
            savename = "".join(methodname.split()).lower()
            self.html[savename] = html

    def write(self, outdir):
        """ Write table to html file """
        for savename, htmlstr in self.html.items():
            with open(os.path.join(outdir, f"{savename}.html"), "w") as f:
                f.write(htmlstr)

    def show(self):
        """ Display table given current parameters"""
        return