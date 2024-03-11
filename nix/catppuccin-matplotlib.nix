{ lib
, buildPythonPackage
, fetchFromGitHub
, poetry-core
, poetry-dynamic-versioning
, pygments
, rich
, setuptools
, matplotlib
}:

buildPythonPackage rec {
  pname = "mplcatppuccin";
  version = "0.4";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "catppuccin";
    repo = "matplotlib";
    rev = "f55eebb6b7deffeb582c1375ba5d1a70287f8c15";
    hash = "sha256-2nkFIeecY6uF3w1xf+iSnQoropUH7aXZ+waX8VgLg9o=";
  };

  nativeBuildInputs = [
    poetry-core
    poetry-dynamic-versioning
    setuptools
    matplotlib
  ];

  passthru.optional-dependencies = {
    pygments = [ pygments ];
    rich = [ rich ];
  };

  pythonImportsCheck = [ "mplcatppuccin" ];

  meta = {
    description = "📊 Soothing pastel theme for matplotlib";
    homepage = "https://github.com/brambozz/matplotlib-catppuccin";
    maintainers = with lib.maintainers; [ cowboylaserkittenjetshark ];
    license = lib.licenses.mit;
  };
}
