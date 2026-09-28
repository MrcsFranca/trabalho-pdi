{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python313.withPackages (ps:
      with ps; [
        numpy
        opencv4
        matplotlib
        textual
        textual-image
        pillow
        rich-pixels
      ]))
  ];
}
