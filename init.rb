# Compatibility patch for Ruby 3.0+ which removed tainted? and untaint methods
# This must be loaded before any gems via require_relative in the Gemfile
class Object
  def tainted?
    false
  end

  def untaint
    self
  end
end
