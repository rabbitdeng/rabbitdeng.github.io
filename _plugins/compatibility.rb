# Compatibility patch for Ruby 3.0+ which removed the tainted? method
class String
  def tainted?
    false
  end
end

class Numeric
  def tainted?
    false
  end
end

class TrueClass
  def tainted?
    false
  end
end

class FalseClass
  def tainted?
    false
  end
end

class NilClass
  def tainted?
    false
  end
end