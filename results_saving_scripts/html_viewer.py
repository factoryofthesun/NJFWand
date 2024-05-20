# Generate HTML table with only the gifs
import numpy as np
import os
import sys
from collections import defaultdict
import base64

"""
The current format is the following table:
             Details | Initial UV | Final UV | UV Opt gif | Initial 360 | Final 360 | All other loss gifs in alphabetical order
Experiment N  {based on the experiment name}           -        -            -
"""
class MeshTableViewer:
    """ Class for visualizing table of mesh results """
    def __init__(self, expnames, descriptions, imgs, extraimgs = None, title = None, tdwidth=100):
        """ expnames: keys for descriptions/imgs dicts
            descriptions: dict(expname: list of bulleted experiment descriptions)
            imgs: dict(expname: [list of (title, img path)])"""
        self.expnames = expnames
        self.descriptions = descriptions
        self.imgs = imgs
        self.extraimgs = extraimgs
        self.title = title
        self.tdwidth = tdwidth

        # Compile unique column set
        self.columns = ['Expname', 'Details']
        imgcolumns = []
        for vallist in imgs.values():
            imgcolumns.extend([val[0] for val in vallist])

        # Make unique while keeping the order
        imgcolumns = list(dict.fromkeys(imgcolumns))
        self.columns += imgcolumns

        # Set HTML string
        self.html = None
        self.sethtml()

    def sethtml(self):
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
        html += (".zoom {\n"
                 "transition: transform .1s;\n"
            "}\n")
        html += (".zoom:hover{\n"
    "transform:scale(2.5);\n"
                "}\n")
        html += ("td {\n"
                "font-size: 0.8vw;\n"
                "}\n")
        html += "</style>\n"
        html += (f"</head>\n"
                f"<body>\n"
                )

        # Add title
        if self.title is not None:
            html += f"<h1>{self.title}</h1>\n"

        # Init table with headers
        html += "<table style='width:100%;table-layout:fixed;'>\n"
        html += "  <tr>\n"

        # Define columns
        for column in self.columns:
            html += f"    <th style='text-align:center;white-space:nowrap;width:{self.tdwidth}%'>{column}</th>\n"
        html += "  </tr>\n"

        # New table row for each experiment
        for expname in self.expnames:
            # Expname
            html += "  <tr>\n"
            html += f"    <td style='text-align:center;word-wrap:break-word'>{expname}</td>\n"

            # Description
            html += f"    <td style='width:{self.tdwidth}%;word-wrap:break-word'>\n"
            html += "       <ul>\n"
            for desc in self.descriptions[expname]:
                html += f"<li>{desc}</li>\n"
            html += "      </ul>\n</td>\n"

            # Imgs
            for title, imgpath in self.imgs[expname]:
                html += f"    <td style='text-align:center;white-space:nowrap;width:{self.tdwidth}%'>\n"
                html += f"      <div class='zoom'><img style='display:block;' width='100%' height='100%' src='{imgpath}' alt='{imgpath}'/></div>\n"
                html += "     </td>\n"
            html += "   </tr>\n"

            ## NOTE: Additional row if set (imgs without zoom)
            if self.extraimgs is not None and expname in self.extraimgs.keys():
                # Expname
                html += "  <tr>\n"
                html += f"    <td style='text-align:center;word-wrap:break-word'></td>\n"

                # Description
                html += f"    <td style='width:{self.tdwidth}%;word-wrap:break-word'>\n"
                html += "       <ul>\n"
                html += "      </ul>\n</td>\n"

                # Imgs
                for title, imgpath in self.extraimgs[expname]:
                    html += f"    <td style='text-align:center;white-space:nowrap;width:{self.tdwidth}%'>\n"
                    html += f"     <img style='display:block;' width='100%' height='100%' src='{imgpath}' alt='{imgpath}'/>\n"
                    html += "     </td>\n"
                html += "   </tr>\n"

        html += " </table>\n"
        html += "</body>\n</html>\n"

        self.html = html

    def write(self, outdir):
        """ Write table to html file """
        with open(outdir, "w") as f:
            f.write(self.html)

    def show(self):
        """ Display table given current parameters"""
        return