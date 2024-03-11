{
  description = "KNN";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs {
          inherit system; config.allowUnfree = true; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }:
      let
        catppuccin-matplotlib = ps: ps.callPackage ./nix/catppuccin-matplotlib.nix {};
        pythonLayered = pkgs.python3.withPackages(ps: with ps; [
          (catppuccin-matplotlib ps)
        ]);
      in  
      {
        default = pkgs.mkShell {
          packages = with pkgs; [ pythonLayered virtualenv ] ++
            (with pkgs.python3Packages;
            [ 
              python-lsp-server
              rope
              pyflakes
              pandas
              scikit-learn
              matplotlib
              seaborn
            ]);
        };
      });
    };
}
